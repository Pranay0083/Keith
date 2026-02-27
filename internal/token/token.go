package token

import (
	"context"
	"fmt"
	"time"

	"github.com/Pranay0083/keith/internal/auth"
	"github.com/Pranay0083/keith/internal/config"
	"golang.org/x/oauth2"
)

// GetValidToken returns a valid access token for the active account.
// For API key accounts, it returns the API key directly.
func GetValidToken() (string, error) {
	acc, err := config.GetActiveAccount()
	if err != nil {
		return "", err
	}

	// API key accounts don't need token refresh
	if acc.Provider == config.ProviderAPIKey {
		return acc.APIKey, nil
	}

	expiry := time.Unix(acc.Expiry, 0)
	if time.Now().Add(5 * time.Minute).Before(expiry) {
		return acc.AccessToken, nil
	}

	// Token is expired or about to expire. Refresh it.
	fmt.Println("Access token expired, refreshing...")

	oauthConfig := auth.GetOAuthConfig(acc.Provider)
	oldToken := &oauth2.Token{
		RefreshToken: acc.RefreshToken,
	}

	tokenSource := oauthConfig.TokenSource(context.Background(), oldToken)
	newToken, err := tokenSource.Token()
	if err != nil {
		return "", fmt.Errorf("failed to refresh token (try 'keith login' again): %v", err)
	}

	// Update config with new token
	updatedAcc := config.Account{
		Email:        acc.Email,
		Provider:     acc.Provider,
		RefreshToken: newToken.RefreshToken,
		AccessToken:  newToken.AccessToken,
		Expiry:       newToken.Expiry.Unix(),
	}
	err = config.AddOrUpdateAccount(updatedAcc)
	if err != nil {
		return "", fmt.Errorf("failed to save refreshed token: %v", err)
	}

	return newToken.AccessToken, nil
}

// GetActiveProvider returns the provider of the active account.
func GetActiveProvider() (config.Provider, error) {
	acc, err := config.GetActiveAccount()
	if err != nil {
		return "", err
	}
	return acc.Provider, nil
}
