package agent

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"time"

	"github.com/Pranay0083/keith/internal/config"
	"golang.org/x/oauth2"
	"google.golang.org/api/calendar/v3"
	"google.golang.org/api/option"
)

// ConnectGoogleCalendarTool brokers a native OAuth popup using the Embedded payload
type ConnectGoogleCalendarTool struct{}

func (t *ConnectGoogleCalendarTool) Name() string { return "connect_google_calendar" }
func (t *ConnectGoogleCalendarTool) Description() string {
	return "Connects a Google Calendar profile natively using the embedded Global Config. Arguments: 'profile' (e.g. 'personal')."
}
func (t *ConnectGoogleCalendarTool) Execute(args map[string]string) (string, error) {
	profile, ok := args["profile"]
	if !ok || profile == "" {
		return "", fmt.Errorf("missing 'profile' argument")
	}

	oauthConfig, err := config.GetOAuthConfig()
	if err != nil {
		return "", err
	}

	authURL := oauthConfig.AuthCodeURL(profile, oauth2.AccessTypeOffline, oauth2.ApprovalForce)

	err = openBrowser(authURL)
	if err != nil {
		return fmt.Sprintf("Failed to open browser natively: %v\nPlease instruct the user to click this URL: %s", err, authURL), nil
	}

	return fmt.Sprintf("Desktop Browser invoked natively! Tell the user to complete Google Sign-in on their screen for profile %s.", profile), nil
}

// ReadGoogleCalendarTool fetches events from the Google endpoint
type ReadGoogleCalendarTool struct{}

func (t *ReadGoogleCalendarTool) Name() string { return "read_google_calendar" }
func (t *ReadGoogleCalendarTool) Description() string {
	return "Reads the next N upcoming events from a connected Google Calendar profile using REST APIs." +
		" Arguments: 'profile' (e.g. 'personal'), 'limit' (e.g. '5')."
}
func (t *ReadGoogleCalendarTool) Execute(args map[string]string) (string, error) {
	profile := args["profile"]
	if profile == "" {
		return "", fmt.Errorf("missing 'profile'")
	}

	oauthConfig, tok, err := config.GetOAuthClient(profile)
	if err != nil {
		return "", err
	}

	ctx := context.Background()
	client := oauthConfig.Client(ctx, tok)
	srv, err := calendar.NewService(ctx, option.WithHTTPClient(client))
	if err != nil {
		return "", err
	}

	tNow := time.Now().Format(time.RFC3339)
	limit := int64(5) // default
	if val, ok := args["limit"]; ok && val != "" {
		fmt.Sscanf(val, "%d", &limit)
	}

	events, err := srv.Events.List("primary").ShowDeleted(false).
		SingleEvents(true).TimeMin(tNow).MaxResults(limit).OrderBy("startTime").Do()
	if err != nil {
		return "", err
	}

	if len(events.Items) == 0 {
		return fmt.Sprintf("No upcoming events found in Google API for profile '%s'.", profile), nil
	}

	output := fmt.Sprintf("Upcoming Events for Google Calendar [%s]:\n", profile)
	for _, item := range events.Items {
		date := item.Start.DateTime
		if date == "" {
			date = item.Start.Date
		}
		output += fmt.Sprintf("- %s (At: %s)\n", item.Summary, date)
	}
	return output, nil
}

// CreateGoogleEventTool generates Google Calendar events with optional Meet logic
type CreateGoogleEventTool struct{}

func (t *CreateGoogleEventTool) Name() string { return "create_google_event" }
func (t *CreateGoogleEventTool) Description() string {
	return "Creates a new event with Google Meet natively in Google Calendar." +
		" Arguments: 'profile', 'title', 'start_time_iso' (RFC3339 e.g. 2026-02-27T15:00:00+05:30), 'end_time_iso'."
}
func (t *CreateGoogleEventTool) Execute(args map[string]string) (string, error) {
	profile := args["profile"]
	title := args["title"]
	startIso := args["start_time_iso"]
	endIso := args["end_time_iso"]

	if profile == "" || title == "" || startIso == "" || endIso == "" {
		return "", fmt.Errorf("missing required structural args")
	}

	oauthConfig, tok, err := config.GetOAuthClient(profile)
	if err != nil {
		return "", err
	}

	ctx := context.Background()
	client := oauthConfig.Client(ctx, tok)
	srv, err := calendar.NewService(ctx, option.WithHTTPClient(client))
	if err != nil {
		return "", err
	}

	event := &calendar.Event{
		Summary: title,
		Start: &calendar.EventDateTime{
			DateTime: startIso,
		},
		End: &calendar.EventDateTime{
			DateTime: endIso,
		},
		ConferenceData: &calendar.ConferenceData{
			CreateRequest: &calendar.CreateConferenceRequest{
				RequestId:             fmt.Sprintf("keith-agent-%d", time.Now().Unix()),
				ConferenceSolutionKey: &calendar.ConferenceSolutionKey{Type: "hangoutsMeet"},
			},
		},
	}

	// Important: ConferenceDataVersion(1) tells Google we want to actively generate a Meeting node
	event, err = srv.Events.Insert("primary", event).ConferenceDataVersion(1).Do()
	if err != nil {
		return "", fmt.Errorf("Google Matrix insertion failed natively: %v", err)
	}

	meetLink := "None"
	if event.ConferenceData != nil && len(event.ConferenceData.EntryPoints) > 0 {
		meetLink = event.ConferenceData.EntryPoints[0].Uri
	}

	return fmt.Sprintf("Native Google Calendar Event successfully spawned!\nLink: %s\nMeet Video URL: %s", event.HtmlLink, meetLink), nil
}

// openBrowser dynamically executes the physical OS command to invoke the Graphical Web Client
func openBrowser(url string) error {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", url).Start()
	case "windows":
		err = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	case "darwin":
		err = exec.Command("open", url).Start()
	default:
		err = fmt.Errorf("unsupported platform natively")
	}
	return err
}
