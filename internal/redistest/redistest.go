// Package redistest gives integration tests a Redis client.
//
// Tests that use it run only with KEITH_IT=1. REDIS_URL defaults to the
// docker-compose.yml Redis. Tests share the server (packages run in
// parallel), so they must use their own random keys and channels and never
// flush the database.
package redistest

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

// DefaultRedisURL is the docker-compose.yml Redis.
const DefaultRedisURL = "redis://localhost:6379/0"

// RedisURL returns REDIS_URL, or DefaultRedisURL when unset.
func RedisURL() string {
	if u := os.Getenv("REDIS_URL"); u != "" {
		return u
	}
	return DefaultRedisURL
}

// Client returns a client for RedisURL, closed when t ends. It skips t
// unless KEITH_IT=1.
func Client(t testing.TB) *redis.Client {
	t.Helper()
	if os.Getenv("KEITH_IT") != "1" {
		t.Skip("integration test: set KEITH_IT=1 (and REDIS_URL, or run docker compose up)")
	}
	opt, err := redis.ParseURL(RedisURL())
	if err != nil {
		t.Fatalf("REDIS_URL: %v", err)
	}
	rdb := redis.NewClient(opt)
	t.Cleanup(func() { _ = rdb.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		t.Fatalf("ping Redis at REDIS_URL: %v", err)
	}
	return rdb
}
