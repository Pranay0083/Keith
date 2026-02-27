package agent

import (
	"fmt"
	"os/exec"
)

// ReadMacCalendarTool interacts physically with macOS Calendar app
type ReadMacCalendarTool struct{}

func (t *ReadMacCalendarTool) Name() string { return "read_mac_calendar" }
func (t *ReadMacCalendarTool) Description() string {
	return "Queries the native macOS Calendar app using AppleScript to retrieve all upcoming scheduled events across all connected accounts (iCloud, Google, Outlook). No arguments required."
}
func (t *ReadMacCalendarTool) Execute(args map[string]string) (string, error) {
	script := `
tell application "Calendar"
	set returnText to ""
	set now to (current date)
	set endDate to now + (30 * days)
	repeat with c in calendars
		try
			set theEvents to (every event of c whose start date ≥ now and start date ≤ endDate)
			if (count of theEvents) > 0 then
				set returnText to returnText & "\n--- Calendar: " & name of c & " ---\n"
				repeat with e in theEvents
					set stringDate to (start date of e as string)
					set returnText to returnText & "- " & summary of e & " (At: " & stringDate & ")\n"
				end repeat
			end if
		end try
	end repeat
	if returnText is "" then return "No upcoming events found natively in macOS Calendar."
	return returnText
end tell
`
	cmd := exec.Command("osascript", "-e", script)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("AppleScript Injection failed natively: %v\nOutput: %s", err, string(out))
	}
	return string(out), nil
}

// CreateMacCalendarEventTool injects events directly into macOS Calendar app
type CreateMacCalendarEventTool struct{}

func (t *CreateMacCalendarEventTool) Name() string { return "create_mac_calendar_event" }
func (t *CreateMacCalendarEventTool) Description() string {
	return "Creates a new event physically in the macOS Calendar app using AppleScript. " +
		"Arguments: 'calendar_name' (e.g. 'Personal' or 'Work', exact name required), 'title', 'start_date' (e.g. 'February 27, 2026 at 3:00:00 PM'), 'end_date' (e.g. 'February 27, 2026 at 4:00:00 PM')."
}
func (t *CreateMacCalendarEventTool) Execute(args map[string]string) (string, error) {
	calName := args["calendar_name"]
	title := args["title"]
	startDate := args["start_date"]
	endDate := args["end_date"]

	if calName == "" || title == "" || startDate == "" || endDate == "" {
		return "", fmt.Errorf("missing required structural schema arguments")
	}

	script := fmt.Sprintf(`
set calName to "%s"
set eventTitle to "%s"
set startDate to date "%s"
set endDate to date "%s"

tell application "Calendar"
	set targetCal to (first calendar whose name is calName)
	tell targetCal
		make new event at end with properties {summary:eventTitle, start date:startDate, end date:endDate}
	end tell
end tell
return "Native macOS Calendar Event successfully materialized into: " & calName
`, calName, title, startDate, endDate)

	cmd := exec.Command("osascript", "-e", script)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("AppleScript Matrix insertion failed natively: %v\nOutput: %s", err, string(out))
	}
	return string(out), nil
}
