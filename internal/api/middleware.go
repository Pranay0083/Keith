package api

import (
	"context"
	"errors"
	"net/http"
	"regexp"
	"runtime/debug"
	"time"

	"github.com/Pranay0083/keith/internal/store"
)

type ctxKey int

const (
	ctxRequestID ctxKey = iota
	ctxUser
	ctxSession
)

var requestIDRe = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

// withRequestID takes a well-formed X-Request-ID or makes one, and echoes
// it on the response.
func (s *Server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if !requestIDRe.MatchString(id) {
			id = randomString(12)
		}
		w.Header().Set("X-Request-ID", id)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxRequestID, id)))
	})
}

func requestID(ctx context.Context) string {
	id, _ := ctx.Value(ctxRequestID).(string)
	return id
}

// statusWriter records the status code. Unwrap lets
// http.ResponseController (and so the WebSocket upgrade) reach the
// underlying writer.
type statusWriter struct {
	http.ResponseWriter
	status int
	wrote  bool
}

func (w *statusWriter) WriteHeader(code int) {
	if !w.wrote {
		w.status, w.wrote = code, true
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.wrote {
		w.status, w.wrote = http.StatusOK, true
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// withLogging logs one line per request. It logs the path, never the
// query, which can hold OAuth codes and state.
func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := s.now()
		sw := &statusWriter{ResponseWriter: w}
		next.ServeHTTP(sw, r)
		status := sw.status
		if !sw.wrote {
			status = http.StatusOK
			if r.Header.Get("Upgrade") != "" {
				status = http.StatusSwitchingProtocols // hijacked by the upgrade
			}
		}
		s.log(r).Info("http request", "method", r.Method, "path", r.URL.Path, "status", status,
			"duration_ms", s.now().Sub(start).Milliseconds())
	})
}

// withRecover turns a panic into a 500 and logs its stack.
func (s *Server) withRecover(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			p := recover()
			if p == nil {
				return
			}
			if p == http.ErrAbortHandler {
				panic(p)
			}
			s.log(r).Error("panic serving request", "panic", p, "stack", string(debug.Stack()))
			if sw, ok := w.(*statusWriter); !ok || !sw.wrote {
				s.writeError(w, r, errInternal)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// cors adds the CORS response headers for WEB_ORIGIN. Retry-After is
// exposed so the web app can read it on a 429.
func (s *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Add("Vary", "Origin")
		if s.allowedOrigin(r) {
			h.Set("Access-Control-Allow-Origin", s.cfg.WebOrigin)
			h.Set("Access-Control-Allow-Credentials", "true")
			h.Set("Access-Control-Expose-Headers", "Retry-After")
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) allowedOrigin(r *http.Request) bool {
	o := r.Header.Get("Origin")
	return o != "" && o == s.cfg.WebOrigin
}

// preflight answers an OPTIONS preflight. Only WEB_ORIGIN gets the
// Access-Control-Allow-* headers.
func preflight(w http.ResponseWriter, r *http.Request) {
	if w.Header().Get("Access-Control-Allow-Origin") != "" {
		h := w.Header()
		h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE")
		h.Set("Access-Control-Allow-Headers", "Content-Type")
		h.Set("Access-Control-Max-Age", "600")
	}
	w.WriteHeader(http.StatusNoContent)
}

// unsafeMethod reports whether a method changes state.
func unsafeMethod(m string) bool {
	switch m {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		return true
	}
	return false
}

// browserAPI wraps a route the web app calls with fetch: CORS, then the
// Origin check on state-changing methods (before anything else touches
// the request), then the body's content type and size.
func (s *Server) browserAPI(next http.Handler) http.Handler {
	return s.cors(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if unsafeMethod(r.Method) && !s.allowedOrigin(r) {
			s.writeError(w, r, errBadOrigin)
			return
		}
		if err := checkBody(w, r); err != nil {
			s.writeError(w, r, err)
			return
		}
		next.ServeHTTP(w, r)
	}))
}

// authed requires a valid keith_session cookie for a user who still
// exists, and puts the user in the context.
func (s *Server) authed(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, cl, err := s.sessionUser(r)
		if err != nil {
			s.writeError(w, r, err)
			return
		}
		ctx := context.WithValue(r.Context(), ctxUser, user)
		ctx = context.WithValue(ctx, ctxSession, cl)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// sessionUser returns the user of the request's session cookie and the
// cookie's claims, or errUnauthenticated: for a missing, invalid or
// expired cookie, a deleted user, or a cookie revoked by logout.
func (s *Server) sessionUser(r *http.Request) (store.User, sessionClaims, error) {
	c, err := r.Cookie(sessionCookie)
	if err != nil {
		return store.User{}, sessionClaims{}, errUnauthenticated
	}
	cl, err := s.signer.verifySession(c.Value, s.now())
	if err != nil {
		return store.User{}, sessionClaims{}, errUnauthenticated
	}
	u, err := s.store.GetUser(r.Context(), cl.UserID)
	if errors.Is(err, store.ErrNotFound) {
		return store.User{}, sessionClaims{}, errUnauthenticated
	}
	if err != nil {
		return store.User{}, sessionClaims{}, err
	}
	if sessionRevoked(u, cl) {
		return store.User{}, sessionClaims{}, errUnauthenticated
	}
	return u, cl, nil
}

// sessionRevoked reports whether a logout revoked the cookie: it was
// issued at or before the user's sessions_valid_after.
func sessionRevoked(u store.User, cl sessionClaims) bool {
	return !u.SessionsValidAfter.IsZero() && !cl.IssuedAt().After(u.SessionsValidAfter)
}

// currentUser returns the signed-in user; only valid behind authed.
func currentUser(r *http.Request) store.User {
	u, _ := r.Context().Value(ctxUser).(store.User)
	return u
}

// currentSession returns the session cookie's claims; only valid behind
// authed.
func currentSession(r *http.Request) sessionClaims {
	cl, _ := r.Context().Value(ctxSession).(sessionClaims)
	return cl
}

func sessionExpiry(r *http.Request) time.Time {
	if cl := currentSession(r); cl.Exp != 0 {
		return time.Unix(cl.Exp, 0)
	}
	return time.Time{}
}
