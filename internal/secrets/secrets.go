// Package secrets encrypts org and repo secrets at rest and redacts their
// values from everything the worker emits.
//
// Threat model, briefly:
//
//   - Sealing. Each value is encrypted with AES-256-GCM under its own random
//     data key (DEK), and the DEK is wrapped with AES-256-GCM under
//     SECRETS_KEY, the key-encryption key (KEK). A copy of the secrets
//     table without the KEK reveals only each value's length and which KEK
//     sealed it. Tampering with any byte of a ciphertext makes Open fail.
//   - The aad. Seal and Open take caller context (the store passes the
//     row's org, repo and name), which is authenticated but not stored. An
//     attacker who can write to the secrets table but lacks the KEK cannot
//     copy one org's ciphertext into another org's row, or swap two names,
//     and have the worker decrypt it there.
//   - Redaction. A Redactor removes secret values from event payloads and
//     streamed terminal output before they are stored or published. It
//     catches the raw value, its base64 (all four alphabets/paddings), URL
//     query and path escapes and JSON string escapes, and a value embedded
//     at any offset in a larger base64 blob (Basic auth headers, encoded
//     configs). It is best-effort defence in depth for the event log and
//     UI, not a boundary: it does not catch hex or other encodings, partial
//     or transformed values (case changes, reversals, a value broken up by
//     ANSI escapes or line wrapping), values shorter than MinRedactLen, or a
//     value split across separate events (half in a tool input and half in
//     its output, or across events not fed through one Stream). Code in the
//     sandbox can read the injected env vars, so anything there can
//     exfiltrate a secret on purpose.
package secrets

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
)

// KeySize is the length in bytes of a key-encryption key (SECRETS_KEY).
// It equals config.SecretsKeySize.
const KeySize = 32

// Ciphertext layout, version 1. All sizes are in bytes; n is the plaintext
// length.
//
//	offset  size  field
//	0       1     format version, 0x01
//	1       4     key ID: SHA-256(keyIDDomain || KEK)[:4], names the KEK
//	5       12    wrap nonce
//	17      48    wrapped DEK: AES-256-GCM(KEK, wrap nonce, DEK), 32 + 16 tag
//	65      12    data nonce
//	77      n+16  data: AES-256-GCM(DEK, data nonce, plaintext), n + 16 tag
//
// The DEK wrap authenticates bytes 0..4 (the header) as additional data.
// The data encryption authenticates bytes 0..76 (everything before the
// data) followed by the caller's aad; the prefix has a fixed length, so the
// concatenation is unambiguous. The total size is n + Overhead.
const (
	formatV1 byte = 1

	versionLen = 1
	keyIDLen   = 4
	nonceLen   = 12
	tagLen     = 16
	dekLen     = 32

	offKeyID      = versionLen                      // 1
	offWrapNonce  = offKeyID + keyIDLen             // 5
	offWrappedDEK = offWrapNonce + nonceLen         // 17
	offDataNonce  = offWrappedDEK + dekLen + tagLen // 65
	offData       = offDataNonce + nonceLen         // 77

	headerLen = offWrapNonce // version + key ID

	// Overhead is how many bytes longer a version 1 ciphertext is than its
	// plaintext.
	Overhead = offData + tagLen // 93
)

// keyIDDomain separates key-ID hashing from any other use of SHA-256 over
// the KEK, so the ID reveals nothing reusable about the key.
const keyIDDomain = "keith/secrets/kek-id/v1\x00"

// Errors returned by NewKeyring, Seal and Open. They never include
// plaintext, aad or key material; wrapped forms add only sizes, versions
// and key IDs.
var (
	// ErrKeySize means a key is not KeySize bytes.
	ErrKeySize = errors.New("secrets: key must be 32 bytes")
	// ErrMalformed means the ciphertext is too short to be valid.
	ErrMalformed = errors.New("secrets: malformed ciphertext")
	// ErrUnknownVersion means the ciphertext's format version byte is not
	// one this package can read.
	ErrUnknownVersion = errors.New("secrets: unknown ciphertext version")
	// ErrUnknownKey means the ciphertext was sealed with a KEK that is not
	// in the keyring.
	ErrUnknownKey = errors.New("secrets: ciphertext sealed with a key not in the keyring")
	// ErrDecrypt means authentication failed: the wrong key, the wrong
	// aad, or a modified ciphertext.
	ErrDecrypt = errors.New("secrets: decryption failed")
)

type keyID [keyIDLen]byte

func (id keyID) String() string { return hex.EncodeToString(id[:]) }

// Keyring seals with a primary KEK and opens with any of its KEKs, so the
// KEK can be rotated: deploy the new key as primary with the old one in
// old, re-seal every row, then drop the old key.
//
// A Keyring is immutable and safe for concurrent use.
type Keyring struct {
	primaryID keyID
	keys      map[keyID][]byte // private copies, never modified
}

// NewKeyring returns a keyring that seals with primary and opens with
// primary or any key in old. Every key must be KeySize bytes. The keys are
// copied, so the caller may clear its slices afterwards. Listing the same
// key twice is allowed; two different keys whose IDs collide (about a
// 1 in 2^32 chance per pair) are rejected, and one of them should be
// regenerated.
func NewKeyring(primary []byte, old ...[]byte) (*Keyring, error) {
	k := &Keyring{keys: make(map[keyID][]byte, 1+len(old))}
	all := append([][]byte{primary}, old...)
	for i, key := range all {
		name := "primary key"
		if i > 0 {
			name = fmt.Sprintf("old key %d", i-1)
		}
		if len(key) != KeySize {
			return nil, fmt.Errorf("%w: %s is %d bytes", ErrKeySize, name, len(key))
		}
		id := kekID(key)
		if prev, ok := k.keys[id]; ok {
			if subtle.ConstantTimeCompare(prev, key) == 1 {
				continue // the same key listed twice
			}
			return nil, fmt.Errorf("secrets: %s has the same key ID (%s) as another key; generate a new key", name, id)
		}
		k.keys[id] = bytes.Clone(key)
		if i == 0 {
			k.primaryID = id
		}
	}
	return k, nil
}

// kekID returns the key ID for a KEK.
func kekID(key []byte) keyID {
	h := sha256.New()
	h.Write([]byte(keyIDDomain))
	h.Write(key)
	var id keyID
	copy(id[:], h.Sum(nil))
	return id
}

// newGCM returns AES-256-GCM with the standard 12-byte nonce and 16-byte
// tag. A fresh AEAD is built per operation; the cost is negligible next to
// a database round trip, and it avoids relying on undocumented concurrency
// guarantees of cipher.AEAD.
func newGCM(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("secrets: aes: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("secrets: gcm: %w", err)
	}
	return gcm, nil
}

// dataAD returns the additional data for the data encryption: the fixed
// 77-byte prefix of the ciphertext followed by the caller's aad.
func dataAD(prefix, aad []byte) []byte {
	ad := make([]byte, 0, len(prefix)+len(aad))
	ad = append(ad, prefix...)
	return append(ad, aad...)
}

// Seal encrypts plaintext under a fresh data key wrapped with the primary
// KEK, binding aad (for example the row's org/repo/name). The same aad
// must be passed to Open. Sealing the same plaintext twice gives different
// ciphertexts.
func (k *Keyring) Seal(plaintext, aad []byte) ([]byte, error) {
	if k == nil {
		return nil, errors.New("secrets: Seal on nil Keyring")
	}
	kek, err := newGCM(k.keys[k.primaryID])
	if err != nil {
		return nil, err
	}

	dek := make([]byte, dekLen)
	defer clear(dek)
	if _, err := rand.Read(dek); err != nil {
		return nil, fmt.Errorf("secrets: generate data key: %w", err)
	}
	dekAEAD, err := newGCM(dek)
	if err != nil {
		return nil, err
	}

	out := make([]byte, offData, Overhead+len(plaintext))
	out[0] = formatV1
	copy(out[offKeyID:], k.primaryID[:])
	if _, err := rand.Read(out[offWrapNonce:offWrappedDEK]); err != nil {
		return nil, fmt.Errorf("secrets: generate nonce: %w", err)
	}
	if _, err := rand.Read(out[offDataNonce:offData]); err != nil {
		return nil, fmt.Errorf("secrets: generate nonce: %w", err)
	}

	// Wrap the DEK into bytes 17..64. The header is copied because
	// cipher.AEAD forbids dst and additional data from overlapping.
	header := bytes.Clone(out[:headerLen])
	kek.Seal(out[offWrappedDEK:offWrappedDEK], out[offWrapNonce:offWrappedDEK], dek, header)

	// Encrypt the value after the prefix, authenticating the prefix and aad.
	ad := dataAD(out[:offData], aad)
	return dekAEAD.Seal(out, out[offDataNonce:offData], plaintext, ad), nil
}

// Open decrypts a ciphertext from Seal, with any key in the keyring. It
// returns ErrMalformed for a truncated input, ErrUnknownVersion for an
// unknown format, ErrUnknownKey if the sealing KEK is not in the keyring,
// and ErrDecrypt for the wrong aad or key, or any modification.
func (k *Keyring) Open(ciphertext, aad []byte) ([]byte, error) {
	if k == nil {
		return nil, errors.New("secrets: Open on nil Keyring")
	}
	if len(ciphertext) < versionLen {
		return nil, fmt.Errorf("%w: empty", ErrMalformed)
	}
	if v := ciphertext[0]; v != formatV1 {
		return nil, fmt.Errorf("%w %d", ErrUnknownVersion, v)
	}
	if len(ciphertext) < Overhead {
		return nil, fmt.Errorf("%w: %d bytes, want at least %d", ErrMalformed, len(ciphertext), Overhead)
	}
	var id keyID
	copy(id[:], ciphertext[offKeyID:offWrapNonce])
	key, ok := k.keys[id]
	if !ok {
		return nil, fmt.Errorf("%w (key ID %s)", ErrUnknownKey, id)
	}

	kek, err := newGCM(key)
	if err != nil {
		return nil, err
	}
	dek, err := kek.Open(nil,
		ciphertext[offWrapNonce:offWrappedDEK],
		ciphertext[offWrappedDEK:offDataNonce],
		ciphertext[:headerLen])
	if err != nil {
		return nil, ErrDecrypt
	}
	defer clear(dek)
	dekAEAD, err := newGCM(dek)
	if err != nil {
		return nil, ErrDecrypt
	}
	plaintext, err := dekAEAD.Open(nil,
		ciphertext[offDataNonce:offData],
		ciphertext[offData:],
		dataAD(ciphertext[:offData], aad))
	if err != nil {
		return nil, ErrDecrypt
	}
	if plaintext == nil {
		plaintext = []byte{}
	}
	return plaintext, nil
}
