package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Provider string

const (
	ProviderAntigravity Provider = "antigravity"
	ProviderAPIKey      Provider = "apikey"
)

type Account struct {
	Email        string   `json:"email"`
	Provider     Provider `json:"provider"`
	RefreshToken string   `json:"refresh_token,omitempty"`
	AccessToken  string   `json:"access_token,omitempty"`
	APIKey       string   `json:"api_key,omitempty"`
	Expiry       int64    `json:"expiry,omitempty"`
}

type AccountsFile struct {
	Accounts         []Account        `json:"accounts"`
	TelegramToken    string           `json:"telegram_token,omitempty"`
	TelegramPassword string           `json:"telegram_password,omitempty"`
	SuperUser        string           `json:"super_user,omitempty"`
	TelegramSessions map[string]int64 `json:"telegram_sessions,omitempty"` // Map of format "ChatID" -> Expiry Unix Timestamp
	GmailAddress     string           `json:"gmail_address,omitempty"`
	GmailAppPassword string           `json:"gmail_app_password,omitempty"`
	DefaultModel     string           `json:"default_model,omitempty"`
}

func GetConfigDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".config", "keith")
	err = os.MkdirAll(dir, 0700)
	if err != nil {
		return "", err
	}
	return dir, nil
}

func GetAccountsPath() (string, error) {
	dir, err := GetConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "accounts.json"), nil
}

func LoadAccounts() (*AccountsFile, error) {
	path, err := GetAccountsPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &AccountsFile{Accounts: []Account{}}, nil
	} else if err != nil {
		return nil, err
	}
	var file AccountsFile
	err = json.Unmarshal(data, &file)
	if err != nil {
		return nil, err
	}
	for i := range file.Accounts {
		if file.Accounts[i].Provider == "" {
			file.Accounts[i].Provider = ProviderAntigravity
		}
	}
	return &file, nil
}

func SaveAccounts(file *AccountsFile) error {
	path, err := GetAccountsPath()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func AddOrUpdateAccount(acc Account) error {
	file, err := LoadAccounts()
	if err != nil {
		return err
	}
	var updatedAccounts []Account
	updatedAccounts = append(updatedAccounts, acc)
	for _, existing := range file.Accounts {
		if existing.Email == acc.Email && existing.Provider == acc.Provider {
			continue
		}
		updatedAccounts = append(updatedAccounts, existing)
	}
	file.Accounts = updatedAccounts
	return SaveAccounts(file)
}

func ClearAccounts() error {
	path, err := GetAccountsPath()
	if err != nil {
		return err
	}
	err = os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

func GetActiveAccount() (*Account, error) {
	file, err := LoadAccounts()
	if err != nil {
		return nil, err
	}
	if len(file.Accounts) == 0 {
		return nil, fmt.Errorf("no accounts found. please run 'keith login' first")
	}
	return &file.Accounts[0], nil
}

func SetActiveAccount(email string) error {
	file, err := LoadAccounts()
	if err != nil {
		return err
	}
	var foundAcc *Account
	var otherAccs []Account
	for _, acc := range file.Accounts {
		if acc.Email == email {
			foundAcc = &acc
		} else {
			otherAccs = append(otherAccs, acc)
		}
	}
	if foundAcc == nil {
		return fmt.Errorf("account with email %s not found", email)
	}
	file.Accounts = append([]Account{*foundAcc}, otherAccs...)
	return SaveAccounts(file)
}
