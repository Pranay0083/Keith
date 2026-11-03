// Package telegram sends messages through the Telegram Bot API.
//
// It is the send half of the old gateway Telegram adapter, kept so V2 can
// wire Telegram in as a notification channel. Nothing calls it in V1.
package telegram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// DefaultBaseURL is the public Telegram Bot API endpoint.
const DefaultBaseURL = "https://api.telegram.org"

// MaxMessageLength is Telegram's limit on the text of one message.
// Callers split longer text themselves.
const MaxMessageLength = 4096

// Client is a minimal Telegram Bot API client. The zero value is not
// usable; construct it with New.
type Client struct {
	token      string
	baseURL    string
	httpClient *http.Client
}

// New returns a Client for the bot identified by token.
func New(token string) *Client {
	return &Client{
		token:      token,
		baseURL:    DefaultBaseURL,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// WithBaseURL returns a copy of c that talks to baseURL instead of the
// public API (used by tests).
func (c *Client) WithBaseURL(baseURL string) *Client {
	cp := *c
	cp.baseURL = strings.TrimRight(baseURL, "/")
	return &cp
}

// WithHTTPClient returns a copy of c that sends requests with hc.
func (c *Client) WithHTTPClient(hc *http.Client) *Client {
	cp := *c
	cp.httpClient = hc
	return &cp
}

// SendMessage posts text as a plain-text message to chatID.
func (c *Client) SendMessage(ctx context.Context, chatID int64, text string) error {
	if text == "" {
		return errors.New("telegram: empty message text")
	}
	body, err := json.Marshal(map[string]any{
		"chat_id": chatID,
		"text":    text,
	})
	if err != nil {
		return fmt.Errorf("telegram: encode request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/bot%s/sendMessage", c.baseURL, c.token)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("telegram: build request: %s", c.redact(err.Error()))
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		// net/http errors include the request URL, which embeds the bot token.
		var uerr *url.Error
		if errors.As(err, &uerr) {
			return fmt.Errorf("telegram: send message: %s", c.redact(uerr.Err.Error()))
		}
		return fmt.Errorf("telegram: send message: %s", c.redact(err.Error()))
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var result struct {
		OK          bool   `json:"ok"`
		Description string `json:"description"`
	}
	_ = json.Unmarshal(raw, &result)
	if resp.StatusCode != http.StatusOK || !result.OK {
		desc := result.Description
		if desc == "" {
			desc = strings.TrimSpace(string(raw))
		}
		return fmt.Errorf("telegram: send message: status %d: %s", resp.StatusCode, c.redact(desc))
	}
	return nil
}

func (c *Client) redact(s string) string {
	if c.token == "" {
		return s
	}
	return strings.ReplaceAll(s, c.token, "[REDACTED]")
}
