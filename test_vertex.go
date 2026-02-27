package main

import (
    "fmt"
    "github.com/Pranay0083/keith/internal/token"
    "bytes"
    "net/http"
    "io"
)

func main() {
    tok, _ := token.GetValidToken()
    req, _ := http.NewRequest("POST", "https://us-central1-aiplatform.googleapis.com/v1/projects/robotic-medium-0zftk/locations/us-central1/publishers/google/models/text-embedding-004:predict", bytes.NewBufferString(`{"instances":[{"content":"hello"}]}`))
    req.Header.Set("Authorization", "Bearer "+tok)
    req.Header.Set("Content-Type", "application/json")
    resp, _ := http.DefaultClient.Do(req)
    body, _ := io.ReadAll(resp.Body)
    fmt.Println(resp.StatusCode)
    fmt.Println(string(body))
}
