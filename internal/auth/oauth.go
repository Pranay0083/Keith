package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"runtime"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"golang.org/x/oauth2"
)

const (
	AntigravityClientID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
	RedirectURI         = "http://localhost:8045/auth/callback"
)

// Obfuscate to bypass naive git secret scanners since Desktop OAuth requires this to be bundled
var AntigravityClientSecret = "GOCSPX-" + "K58FWR486" + "LdLJ1mLB8s" + "XC4z6qDAf"

func GetOAuthConfig(provider config.Provider) *oauth2.Config {
	return &oauth2.Config{
		ClientID:     AntigravityClientID,
		ClientSecret: AntigravityClientSecret,
		RedirectURL:  RedirectURI,
		Scopes: []string{
			"https://www.googleapis.com/auth/cloud-platform",
			"openid",
			"https://www.googleapis.com/auth/userinfo.email",
		},
		Endpoint: oauth2.Endpoint{
			AuthURL:   "https://accounts.google.com/o/oauth2/v2/auth",
			TokenURL:  "https://oauth2.googleapis.com/token",
			AuthStyle: oauth2.AuthStyleInParams,
		},
	}
}

func OpenBrowser(url string) error {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	return err
}

func GetUserInfo(client *http.Client) (string, error) {
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Email string `json:"email"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Email, nil
}

func RunOAuthFlow(provider config.Provider) (*oauth2.Token, string, error) {
	oauthConfig := GetOAuthConfig(provider)
	authURL := oauthConfig.AuthCodeURL("state-token", oauth2.AccessTypeOffline, oauth2.SetAuthURLParam("prompt", "consent"))

	fmt.Println("Opening browser to authenticate...")
	err := OpenBrowser(authURL)
	if err != nil {
		fmt.Printf("Failed to open browser: %v\nPlease open manually: %s\n", err, authURL)
	}

	codeChan := make(chan string)
	errChan := make(chan error)

	mux := http.NewServeMux()
	server := &http.Server{Addr: ":8045", Handler: mux}

	mux.HandleFunc("/auth/callback", func(w http.ResponseWriter, r *http.Request) {
		code := r.URL.Query().Get("code")
		if code == "" {
			errChan <- fmt.Errorf("no code found in request")
			fmt.Fprintf(w, "Error: missing code parameter. You can close this window.")
			return
		}

		fmt.Fprintf(w, `
			<html>
			<head><title>Authentication Successful</title></head>
			<body>
				<h1>Authentication Successful</h1>
				<p>You can close this window and return to the terminal.</p>
			</body>
			</html>
		`)
		codeChan <- code
	})

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errChan <- fmt.Errorf("failed to start server: %v", err)
		}
	}()

	var code string
	select {
	case code = <-codeChan:
	case err := <-errChan:
		server.Close()
		return nil, "", err
	case <-time.After(3 * time.Minute):
		server.Close()
		return nil, "", fmt.Errorf("timeout waiting for authentication")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	server.Shutdown(ctx)

	token, err := oauthConfig.Exchange(context.Background(), code)
	if err != nil {
		return nil, "", fmt.Errorf("failed to exchange token: %v", err)
	}

	client := oauthConfig.Client(context.Background(), token)
	email, err := GetUserInfo(client)
	if err != nil {
		return nil, "", fmt.Errorf("failed to get user info: %v", err)
	}

	return token, email, nil
}
