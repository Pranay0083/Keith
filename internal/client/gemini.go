package client

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	GeminiAPIEndpoint = "https://generativelanguage.googleapis.com/v1beta"
)

func GenerateChatGemini(contents []map[string]interface{}, modelID string, systemInstruction string, tools []map[string]interface{}, token string, isAPIKey bool) (string, error) {
	// Build the standard Gemini API request
	requestBody := map[string]interface{}{
		"contents": contents,
		"generationConfig": map[string]interface{}{
			"temperature":     0.4,
			"topP":            1,
			"topK":            40,
			"candidateCount":  1,
			"maxOutputTokens": 8192,
		},
	}

	if systemInstruction != "" {
		requestBody["system_instruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": systemInstruction},
			},
		}
	}

	if len(tools) > 0 {
		requestBody["tools"] = []map[string]interface{}{
			{"function_declarations": tools},
		}
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	// Build URL with model name
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", GeminiAPIEndpoint, modelID)
	if isAPIKey {
		url += "&key=" + token
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	if !isAPIKey {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	httpClient := &http.Client{Timeout: 2 * time.Minute}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
	}

	return parseGeminiSSE(resp.Body)
}

// parseGeminiSSE parses the SSE stream from the standard Gemini API.
func parseGeminiSSE(body io.Reader) (string, error) {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	var fullText strings.Builder
	firstChunk := true

	for scanner.Scan() {
		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")
		if dataStr == "[DONE]" {
			break
		}

		var event map[string]interface{}
		if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
			continue
		}

		candidates, ok := event["candidates"].([]interface{})
		if !ok {
			continue
		}

		for _, c := range candidates {
			candidate, ok := c.(map[string]interface{})
			if !ok {
				continue
			}
			content, ok := candidate["content"].(map[string]interface{})
			if !ok {
				continue
			}
			parts, ok := content["parts"].([]interface{})
			if !ok {
				continue
			}
			for _, p := range parts {
				part, ok := p.(map[string]interface{})
				if !ok {
					continue
				}

				// Handle standard text response
				if text, ok := part["text"].(string); ok && text != "" {
					if firstChunk {
						fmt.Print("\n")
						firstChunk = false
					}
					fmt.Print(text)
					fullText.WriteString(text)
				}

				// Handle native Function Calling response
				if funcCall, ok := part["functionCall"].(map[string]interface{}); ok {
					name, _ := funcCall["name"].(string)
					argsMap, _ := funcCall["args"].(map[string]interface{})

					// Synthesize the markdown JSON block that loop.go expects
					type ToolCall struct {
						ToolName string            `json:"tool_name"`
						Args     map[string]string `json:"args"`
					}

					strArgs := make(map[string]string)
					for k, v := range argsMap {
						strArgs[k] = fmt.Sprintf("%v", v)
					}

					tc := ToolCall{
						ToolName: name,
						Args:     strArgs,
					}

					jsonBytes, _ := json.MarshalIndent(tc, "", "  ")
					synthStr := fmt.Sprintf("```json\n%s\n```", string(jsonBytes))

					if firstChunk {
						fmt.Print("\n")
						firstChunk = false
					}
					fmt.Println("\n[Native Function Call Intercepted]")
					fullText.WriteString(synthStr)
				}
			}
		}
	}

	if !firstChunk {
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		return fullText.String(), fmt.Errorf("error reading stream: %v", err)
	}

	result := fullText.String()
	if result == "" {
		return "", fmt.Errorf("no text content found in response")
	}

	return result, nil
}
