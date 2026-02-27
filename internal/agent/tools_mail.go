package agent

import (
	"bytes"
	"fmt"
	"net/smtp"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/Pranay0083/keith/internal/config"
)

// SaveGmailCredentialsTool securely saves the user's Gmail and App Password to accounts.json
type SaveGmailCredentialsTool struct{}

func (s *SaveGmailCredentialsTool) Name() string { return "save_gmail_credentials" }
func (s *SaveGmailCredentialsTool) Description() string {
	return "If the user asks to integrate their Gmail, organically ask them to generate a 16-character Google App Password (found in Google Account -> Security -> 2-Step Verification). Once they provide their email and App Password, use this exact tool to securely write those credentials into the host config matrix so you possess permanent email capabilities."
}
func (s *SaveGmailCredentialsTool) Execute(args map[string]string) (string, error) {
	email, ok := args["email"]
	if !ok || email == "" {
		return "", fmt.Errorf("missing 'email' argument")
	}
	password, ok := args["app_password"]
	if !ok || password == "" {
		return "", fmt.Errorf("missing 'app_password' argument")
	}

	accs, err := config.LoadAccounts()
	if err != nil {
		return "", err
	}

	accs.GmailAddress = email
	accs.GmailAppPassword = password
	config.SaveAccounts(accs)

	return "SUCCESS! Gmail App Credentials actively secured in ~/.config/keith/accounts.json. You now natively possess the `send_email` and `read_emails` architectural tools!", nil
}

// SendEmailTool securely triggers an SMTP TLS transmission using Google's relay.
type SendEmailTool struct{}

func (s *SendEmailTool) Name() string { return "send_email" }
func (s *SendEmailTool) Description() string {
	return "Transmits an email natively over SMTP on behalf of the user. Only works if the user has already onboarded their App Credentials."
}
func (s *SendEmailTool) Execute(args map[string]string) (string, error) {
	to, ok := args["to"]
	if !ok || to == "" {
		return "", fmt.Errorf("missing 'to' argument")
	}
	subject, ok := args["subject"]
	if !ok || subject == "" {
		return "", fmt.Errorf("missing 'subject' argument")
	}
	body, ok := args["body"]
	if !ok || body == "" {
		return "", fmt.Errorf("missing 'body' argument")
	}

	accs, err := config.LoadAccounts()
	if err != nil || accs.GmailAddress == "" || accs.GmailAppPassword == "" {
		return "", fmt.Errorf("gmail is not configured! Please ask the user to provide their Gmail Address and a Google App Password so you can run save_gmail_credentials first")
	}

	auth := smtp.PlainAuth("", accs.GmailAddress, accs.GmailAppPassword, "smtp.gmail.com")

	contentType := "text/plain; charset=\"UTF-8\""
	if strings.Contains(strings.ToLower(body), "<html>") || strings.Contains(strings.ToLower(body), "<body") {
		contentType = "text/html; charset=\"UTF-8\""
	}

	headers := fmt.Sprintf("To: %s\r\nSubject: %s\r\nMIME-Version: 1.0\r\nContent-Type: %s\r\n\r\n", to, subject, contentType)
	msgBytes := []byte(headers + body)

	err = smtp.SendMail("smtp.gmail.com:587", auth, accs.GmailAddress, []string{to}, msgBytes)
	if err != nil {
		return "", fmt.Errorf("SMTP Transmission Failed: %v", err)
	}

	return fmt.Sprintf("Successfully dispatched email to %s", to), nil
}

// ReadEmailsTool securely spawns a Python IMAP subshell to bypass missing IMAP in Go standard library.
type ReadEmailsTool struct{}

func (s *ReadEmailsTool) Name() string { return "read_emails" }
func (s *ReadEmailsTool) Description() string {
	return "Hooks into Gmail via IMAP returning the JSON content of the most recent N emails from the inbox. Useful for reading notifications or checking mail organically."
}
func (s *ReadEmailsTool) Execute(args map[string]string) (string, error) {
	limitStr, ok := args["limit"]
	if !ok || limitStr == "" {
		limitStr = "5"
	}
	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit <= 0 {
		limit = 5
	}

	accs, err := config.LoadAccounts()
	if err != nil || accs.GmailAddress == "" || accs.GmailAppPassword == "" {
		return "", fmt.Errorf("gmail is not natively configured! Ask the user for their Google App Password so you can run 'save_gmail_credentials' first")
	}

	pyScript := `
import imaplib, email, json, sys

def fetch_emails(user, pwd, limit):
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(user, pwd)
        mail.select('inbox')
        _, search_data = mail.search(None, 'ALL')
        mail_ids = search_data[0].split()
        latest = mail_ids[-limit:]
        latest.reverse()
        result = []
        for m_id in latest:
            _, data = mail.fetch(m_id, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            subject = str(email.header.make_header(email.header.decode_header(msg['Subject']))) if msg['Subject'] else "No Subject"
            sender = msg.get('From', 'Unknown')
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode('utf-8', 'ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode('utf-8', 'ignore')
            result.append({"subject": subject, "from": sender, "snippet": body[:800]})
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    fetch_emails(sys.argv[1], sys.argv[2], int(sys.argv[3]))
`

	scriptPath := "/tmp/keith_imap.py"
	os.WriteFile(scriptPath, []byte(pyScript), 0644)
	defer os.Remove(scriptPath)

	cmd := exec.Command("python3", scriptPath, accs.GmailAddress, accs.GmailAppPassword, fmt.Sprintf("%d", limit))
	var out bytes.Buffer
	var errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut

	err = cmd.Run()
	if err != nil {
		return "", fmt.Errorf("IMAP Sub-Shell Failed: %v\nStderr: %s", err, errOut.String())
	}

	return fmt.Sprintf("Retrieved last %d emails:\n%s", limit, out.String()), nil
}
