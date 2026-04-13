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

	"github.com/Pranay0083/keith/internal/models"
	"github.com/Pranay0083/keith/internal/token"
)

const (
	// The actual Antigravity API endpoint (discovered from community implementations)
	APIEndpoint = "https://daily-cloudcode-pa.sandbox.googleapis.com"
	// Endpoint to discover the project ID tied to the authenticated account
	LoadCodeAssistPath = "/v1internal:loadCodeAssist"
	// Streaming inference endpoint
	StreamGeneratePath = "/v1internal:streamGenerateContent?alt=sse"
)

// discoverProject discovers (or provisions) the cloudaicompanionProject ID for the authenticated user.
func discoverProject(accessToken string) (string, error) {
	reqBody := map[string]interface{}{
		"metadata": map[string]string{
			"ideType":    "ANTIGRAVITY",
			"platform":   "PLATFORM_UNSPECIFIED",
			"pluginType": "GEMINI",
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", APIEndpoint+LoadCodeAssistPath, bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "antigravity/1.20.5 darwin/arm64")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("loadCodeAssist request failed: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("loadCodeAssist failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("failed to parse loadCodeAssist response: %v", err)
	}

	// If we already have a project ID, return it
	if projectID, ok := result["cloudaicompanionProject"].(string); ok && projectID != "" {
		return projectID, nil
	}

	// No project ID — need to onboard. Find the default tier.
	tierID := "free-tier"
	if tiers, ok := result["allowedTiers"].([]interface{}); ok {
		for _, t := range tiers {
			tier, ok := t.(map[string]interface{})
			if !ok {
				continue
			}
			if isDefault, ok := tier["isDefault"].(bool); ok && isDefault {
				if id, ok := tier["id"].(string); ok {
					tierID = id
				}
				break
			}
		}
	}

	fmt.Printf("  No project found, onboarding with tier '%s'...\n", tierID)
	return onboardUser(accessToken, tierID)
}

// onboardUser provisions a new project for the user and polls until complete.
func onboardUser(accessToken string, tierID string) (string, error) {
	reqBody := map[string]interface{}{
		"tierId": tierID,
		"metadata": map[string]string{
			"ideType":    "ANTIGRAVITY",
			"platform":   "PLATFORM_UNSPECIFIED",
			"pluginType": "GEMINI",
		},
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	maxAttempts := 5
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		req, err := http.NewRequest("POST", APIEndpoint+"/v1internal:onboardUser", bytes.NewBuffer(data))
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+accessToken)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "antigravity/1.20.5 darwin/arm64")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", fmt.Errorf("onboardUser request failed: %v", err)
		}

		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("onboardUser failed (status %d): %s", resp.StatusCode, string(body))
		}

		var result map[string]interface{}
		if err := json.Unmarshal(body, &result); err != nil {
			return "", fmt.Errorf("failed to parse onboardUser response: %v", err)
		}

		// Check if operation is done
		done, _ := result["done"].(bool)
		if done {
			return extractProjectID(result)
		}

		fmt.Printf("  Onboarding in progress (attempt %d/%d)...\n", attempt, maxAttempts)
		time.Sleep(2 * time.Second)
	}

	return "", fmt.Errorf("onboarding timed out after %d attempts", maxAttempts)
}

func extractProjectID(result map[string]interface{}) (string, error) {
	// Try response.cloudaicompanionProject
	if resp, ok := result["response"].(map[string]interface{}); ok {
		if proj, ok := resp["cloudaicompanionProject"].(string); ok && proj != "" {
			return proj, nil
		}
		if projObj, ok := resp["cloudaicompanionProject"].(map[string]interface{}); ok {
			if id, ok := projObj["id"].(string); ok && id != "" {
				return id, nil
			}
		}
	}
	// Try top-level cloudaicompanionProject
	if proj, ok := result["cloudaicompanionProject"].(string); ok && proj != "" {
		return proj, nil
	}
	if projObj, ok := result["cloudaicompanionProject"].(map[string]interface{}); ok {
		if id, ok := projObj["id"].(string); ok && id != "" {
			return id, nil
		}
	}

	raw, _ := json.MarshalIndent(result, "", "  ")
	return "", fmt.Errorf("could not extract project ID from onboard response: %s", string(raw))
}

// FetchAvailableModels returns the raw JSON of available models from the API.
func FetchAvailableModels() (string, error) {
	tok, err := token.GetValidToken()
	if err != nil {
		return "", err
	}

	projectID, err := discoverProject(tok)
	if err != nil {
		return "", err
	}

	reqBody, _ := json.Marshal(map[string]interface{}{"project": projectID})
	req, err := http.NewRequest("POST", APIEndpoint+"/v1internal:fetchAvailableModels", bytes.NewBuffer(reqBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "antigravity/1.20.5 darwin/arm64")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetchAvailableModels failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result interface{}
	json.Unmarshal(body, &result)
	pretty, _ := json.MarshalIndent(result, "", "  ")
	return string(pretty), nil
}

func GenerateChat(contents []map[string]interface{}, modelID string, thinkingLevel string, systemInstruction string, tools []map[string]interface{}) (string, error) {
	tok, err := token.GetValidToken()
	if err != nil {
		return "", err
	}

	// Step 1: Discover the project ID
	fmt.Println("  Discovering project ID...")
	projectID, err := discoverProject(tok)
	if err != nil {
		return "", fmt.Errorf("failed to discover project: %v", err)
	}
	fmt.Printf("  Project: %s\n", projectID)

	// Step 2: Build the inference request in the Antigravity format
	model := models.GetModel(modelID)

	genConfig := map[string]interface{}{
		"temperature":     0.4,
		"topP":            1,
		"topK":            40,
		"candidateCount":  1,
		"maxOutputTokens": 8192,
	}

	// Add thinking config if specified
	if thinkingLevel != "" {
		budgets := map[string]int{
			"low":  1024,
			"high": 16384,
		}
		budget, ok := budgets[thinkingLevel]
		if !ok {
			budget = 1024
		}
		genConfig["thinkingConfig"] = map[string]interface{}{
			"includeThoughts": true,
			"thinkingBudget":  budget,
		}
		fmt.Printf("  Thinking: %s (budget: %d tokens)\n", thinkingLevel, budget)
	}

	reqBody := map[string]interface{}{
		"contents":         contents,
		"generationConfig": genConfig,
	}

	if systemInstruction != "" {
		reqBody["system_instruction"] = map[string]interface{}{
			"parts": []map[string]interface{}{
				{"text": systemInstruction},
			},
		}
	}

	if len(tools) > 0 {
		reqBody["tools"] = []map[string]interface{}{
			{"function_declarations": tools},
		}
	}

	geminiRequest := map[string]interface{}{
		"project":     projectID,
		"requestId":   fmt.Sprintf("keith-%d", time.Now().Unix()),
		"model":       model.ID,
		"userAgent":   "antigravity/1.20.5 darwin/arm64",
		"requestType": "agent",
		"request":     reqBody,
	}

	data, err := json.Marshal(geminiRequest)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", APIEndpoint+StreamGeneratePath, bytes.NewBuffer(data))
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+tok)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "antigravity/1.20.5 darwin/arm64")

	httpClient := &http.Client{Timeout: 2 * time.Minute}
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Step 3: Parse the SSE stream and extract text content
	return parseSSEResponse(resp.Body)
}

// parseSSEResponse reads a Server-Sent Events stream and extracts text from Gemini responses.
func parseSSEResponse(body io.Reader) (string, error) {
	scanner := bufio.NewScanner(body)
	// Increase buffer size for large SSE events
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
			continue // skip non-JSON lines
		}

		// The response may be nested under "response" key
		responseData := event
		if resp, ok := event["response"].(map[string]interface{}); ok {
			responseData = resp
		}

		// Extract text from candidates[].content.parts[].text
		candidates, ok := responseData["candidates"].([]interface{})
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
				// Skip thinking parts
				if thought, exists := part["thought"]; exists {
					if t, ok := thought.(bool); ok && t {
						continue
					}
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

	// Print final newline
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
