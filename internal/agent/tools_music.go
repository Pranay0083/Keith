package agent

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"regexp"
	"strings"
)

// ControlMusicPlaybackTool natively commands the OS Media Players
type ControlMusicPlaybackTool struct{}

func (t *ControlMusicPlaybackTool) Name() string { return "control_music_playback" }
func (t *ControlMusicPlaybackTool) Description() string {
	return "Controls playback state for Spotify. Arguments: 'action' (string: 'play', 'pause', 'playpause', 'next track', 'previous track')."
}
func (t *ControlMusicPlaybackTool) Execute(args map[string]string) (string, error) {
	action := strings.ToLower(args["action"])

	if action == "" {
		return "", fmt.Errorf("missing 'action' argument")
	}

	// Validate action to prevent AppleScript injection attacks
	validActions := map[string]bool{"play": true, "pause": true, "playpause": true, "next track": true, "previous track": true}
	if !validActions[action] {
		return "", fmt.Errorf("invalid action: %s", action)
	}

	script := fmt.Sprintf(`tell application "Spotify" to %s`, action)
	err := exec.Command("osascript", "-e", script).Run()
	if err != nil {
		return "", fmt.Errorf("failed to command Spotify natively: %v", err)
	}

	return fmt.Sprintf("Successfully dispatched '%s' command natively to Spotify.", action), nil
}

// GetCurrentTrackTool natively reads the physical OS playback memory
type GetCurrentTrackTool struct{}

func (t *GetCurrentTrackTool) Name() string { return "get_current_track" }
func (t *GetCurrentTrackTool) Description() string {
	return "Reads the live OS memory to extract what song is currently playing on Spotify."
}
func (t *GetCurrentTrackTool) Execute(args map[string]string) (string, error) {
	script := `
		tell application "Spotify"
			if player state is playing then
				set trackName to name of current track
				set trackArtist to artist of current track
				return trackArtist & " - " & trackName
			else
				return "Nothing is currently playing on Spotify."
			end if
		end tell
	`

	out, err := exec.Command("osascript", "-e", script).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to read Spotify playback memory: %s", string(out))
	}

	return fmt.Sprintf("🎵 Live Playback Telemetry: %s", strings.TrimSpace(string(out))), nil
}

func findSpotifyURIViaDuckDuckGo(query string) string {
	encodedQuery := url.QueryEscape("site:open.spotify.com/track " + query)
	searchURL := "https://html.duckduckgo.com/html/?q=" + encodedQuery

	client := &http.Client{}
	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}

	re := regexp.MustCompile(`open\.spotify\.com/track/([a-zA-Z0-9]+)`)
	matches := re.FindStringSubmatch(string(body))
	if len(matches) > 1 {
		return "spotify:track:" + matches[1]
	}
	return ""
}

// PlaySpecificSongTool natively searches and plays a specific track
type PlaySpecificSongTool struct{}

func (t *PlaySpecificSongTool) Name() string { return "play_specific_song" }
func (t *PlaySpecificSongTool) Description() string {
	return "Forces playback of a specific track on Spotify using natural language search. Arguments: 'query' (e.g. 'Roz Roz The Yellow Diary'). IMPORTANT: Always include the Artist Name in the query to guarantee Spotify plays the correct requested track."
}
func (t *PlaySpecificSongTool) Execute(args map[string]string) (string, error) {
	query := args["query"]

	if query == "" {
		return "", fmt.Errorf("missing 'query' argument")
	}

	trackURI := findSpotifyURIViaDuckDuckGo(query)
	var script string

	if trackURI != "" {
		script = fmt.Sprintf(`tell application "Spotify" to play track "%s"`, trackURI)
	} else {
		script = fmt.Sprintf(`tell application "Spotify" to play track "spotify:search:%s"`, query)
	}

	err := exec.Command("osascript", "-e", script).Run()
	if err != nil {
		return "", fmt.Errorf("Spotify exact URI injection failed: %v", err)
	}

	if trackURI != "" {
		return fmt.Sprintf("Intercepted Spotify matrix and injected pure deterministic stream for: %s (Resolved from '%s')", trackURI, query), nil
	}
	return fmt.Sprintf("Intercepted Spotify matrix and injected fallback stream for: %s", query), nil
}
