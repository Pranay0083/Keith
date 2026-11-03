package anthropic

import "github.com/Pranay0083/keith/internal/llm"

// invalidRequest reports a request this provider can't send.
func invalidRequest(msg string) *llm.Error {
	return &llm.Error{Provider: Name, Kind: llm.ErrInvalidRequest, Message: msg}
}
