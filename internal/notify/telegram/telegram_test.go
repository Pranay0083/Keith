package telegram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSendMessage(t *testing.T) {
	var gotPath string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Errorf("decode body: %v", err)
		}
		w.Write([]byte(`{"ok":true,"result":{}}`))
	}))
	defer srv.Close()

	c := New("123:secret").WithBaseURL(srv.URL)
	if err := c.SendMessage(context.Background(), 42, "hello"); err != nil {
		t.Fatalf("SendMessage: %v", err)
	}
	if gotPath != "/bot123:secret/sendMessage" {
		t.Errorf("path = %q", gotPath)
	}
	if gotBody["chat_id"] != float64(42) || gotBody["text"] != "hello" {
		t.Errorf("body = %v", gotBody)
	}
}

func TestSendMessageAPIErrorRedactsToken(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"ok":false,"description":"Bad Request: chat not found for 123:secret"}`))
	}))
	defer srv.Close()

	err := New("123:secret").WithBaseURL(srv.URL).SendMessage(context.Background(), 1, "x")
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "secret") {
		t.Errorf("error leaks token: %v", err)
	}
	if !strings.Contains(err.Error(), "chat not found") {
		t.Errorf("error lacks description: %v", err)
	}
}

func TestSendMessageTransportErrorRedactsToken(t *testing.T) {
	err := New("123:secret").WithBaseURL("http://127.0.0.1:1").SendMessage(context.Background(), 1, "x")
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), "secret") {
		t.Errorf("error leaks token: %v", err)
	}
}

func TestSendMessageEmpty(t *testing.T) {
	if err := New("t").SendMessage(context.Background(), 1, ""); err == nil {
		t.Fatal("expected error for empty text")
	}
}
