package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	_ "embed"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/calendar/v3"
)

//go:embed secrets/credentials.json
var embeddedCreds []byte

// GetOAuthConfig reads the embedded Google Cloud structural payload
func GetOAuthConfig() (*oauth2.Config, error) {
	// Requesting native read/write Calendar scope natively
	oauthConfig, err := google.ConfigFromJSON(embeddedCreds, calendar.CalendarScope)
	if err != nil {
		return nil, fmt.Errorf("unable to parse embedded client secret file to native config matrix: %v", err)
	}
	// Route back to Keith's core HTTP Gateway
	oauthConfig.RedirectURL = "http://localhost:8080/oauth2callback"
	return oauthConfig, nil
}

// GetTokenFile derives the physical filesystem path for a target identity profile
func GetTokenFile(profile string) (string, error) {
	dir, err := GetConfigDir()
	if err != nil {
		return "", err
	}

	tokensDir := filepath.Join(dir, "gcal_tokens")
	os.MkdirAll(tokensDir, 0755)

	return filepath.Join(tokensDir, fmt.Sprintf("%s_token.json", profile)), nil
}

// SaveToken serializes the Google session key into the local matrix
func SaveToken(profile string, token *oauth2.Token) error {
	path, err := GetTokenFile(profile)
	if err != nil {
		return err
	}

	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	return json.NewEncoder(f).Encode(token)
}

// LoadToken retrieves an active Google session key for a profile
func LoadToken(profile string) (*oauth2.Token, error) {
	path, err := GetTokenFile(profile)
	if err != nil {
		return nil, err
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("no authorization token found natively for profile '%s'. Please instruct the user to run the Connect Calendar setup first", profile)
	}
	defer f.Close()

	tok := &oauth2.Token{}
	err = json.NewDecoder(f).Decode(tok)
	return tok, err
}

// GetOAuthClient Native Client constructor
func GetOAuthClient(profile string) (*oauth2.Config, *oauth2.Token, error) {
	tok, err := LoadToken(profile)
	if err != nil {
		return nil, nil, err
	}

	oauthConfig, err := GetOAuthConfig()
	if err != nil {
		return nil, nil, err
	}
	return oauthConfig, tok, nil
}
