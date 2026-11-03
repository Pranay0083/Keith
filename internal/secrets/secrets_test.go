package secrets

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"strings"
	"sync"
	"testing"
)

func newKey(t *testing.T) []byte {
	t.Helper()
	k := make([]byte, KeySize)
	if _, err := rand.Read(k); err != nil {
		t.Fatal(err)
	}
	return k
}

func newKeyring(t *testing.T, primary []byte, old ...[]byte) *Keyring {
	t.Helper()
	k, err := NewKeyring(primary, old...)
	if err != nil {
		t.Fatalf("NewKeyring: %v", err)
	}
	return k
}

func seal(t *testing.T, k *Keyring, plaintext, aad []byte) []byte {
	t.Helper()
	ct, err := k.Seal(plaintext, aad)
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}
	return ct
}

var testAAD = []byte("org_1/repo_2/DATABASE_URL")

func TestSealOpenRoundTrip(t *testing.T) {
	k := newKeyring(t, newKey(t))
	big := make([]byte, 64<<10)
	rand.Read(big)
	for _, tc := range []struct {
		name      string
		plaintext []byte
		aad       []byte
	}{
		{"empty", nil, testAAD},
		{"empty non-nil", []byte{}, testAAD},
		{"short", []byte("x"), testAAD},
		{"typical", []byte("postgres://u:p@db/keith?sslmode=require"), testAAD},
		{"binary", []byte{0, 1, 2, 0xff, 0xfe}, testAAD},
		{"large", big, testAAD},
		{"no aad", []byte("hunter2hunter2"), nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ct := seal(t, k, tc.plaintext, tc.aad)
			if got, want := len(ct), Overhead+len(tc.plaintext); got != want {
				t.Errorf("len(ciphertext) = %d, want %d", got, want)
			}
			if len(tc.plaintext) >= 4 && bytes.Contains(ct, tc.plaintext) {
				t.Error("ciphertext contains the plaintext")
			}
			pt, err := k.Open(ct, tc.aad)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			if pt == nil || !bytes.Equal(pt, tc.plaintext) {
				t.Errorf("Open = %q, want %q", pt, tc.plaintext)
			}
		})
	}
}

func TestLayout(t *testing.T) {
	key := newKey(t)
	k := newKeyring(t, key)
	ct := seal(t, k, []byte("value"), testAAD)
	if ct[0] != formatV1 {
		t.Errorf("version byte = %d, want %d", ct[0], formatV1)
	}
	id := kekID(key)
	if !bytes.Equal(ct[offKeyID:offWrapNonce], id[:]) {
		t.Errorf("key ID = %x, want %x", ct[offKeyID:offWrapNonce], id)
	}
	if bytes.Equal(id[:], key[:keyIDLen]) {
		t.Error("key ID is a prefix of the key")
	}
	if offData != 77 || Overhead != 93 {
		t.Errorf("offData, Overhead = %d, %d; want 77, 93 (the documented layout)", offData, Overhead)
	}
}

func TestSealIsRandomized(t *testing.T) {
	k := newKeyring(t, newKey(t))
	a := seal(t, k, []byte("same plaintext"), testAAD)
	b := seal(t, k, []byte("same plaintext"), testAAD)
	if bytes.Equal(a, b) {
		t.Fatal("two seals of the same plaintext are identical")
	}
	for _, r := range []struct {
		name     string
		from, to int
	}{
		{"wrap nonce", offWrapNonce, offWrappedDEK},
		{"wrapped DEK", offWrappedDEK, offDataNonce},
		{"data nonce", offDataNonce, offData},
		{"data", offData, len(a)},
	} {
		if bytes.Equal(a[r.from:r.to], b[r.from:r.to]) {
			t.Errorf("%s is the same in both ciphertexts", r.name)
		}
	}
}

func TestOpenWrongAAD(t *testing.T) {
	k := newKeyring(t, newKey(t))
	ct := seal(t, k, []byte("s3cret-value"), testAAD)
	for _, aad := range [][]byte{
		nil,
		[]byte("org_1/repo_2/DATABASE_UR"),
		[]byte("org_9/repo_2/DATABASE_URL"),
		append(bytes.Clone(testAAD), 0),
	} {
		if _, err := k.Open(ct, aad); !errors.Is(err, ErrDecrypt) {
			t.Errorf("Open with aad %q: err = %v, want ErrDecrypt", aad, err)
		}
	}
}

func TestOpenWrongKey(t *testing.T) {
	k1 := newKeyring(t, newKey(t))
	key2 := newKey(t)
	k2 := newKeyring(t, key2)
	ct := seal(t, k1, []byte("s3cret-value"), testAAD)

	if _, err := k2.Open(ct, testAAD); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("Open with another keyring: err = %v, want ErrUnknownKey", err)
	}

	// Point the key ID at k2's key, as a key-ID collision would: the
	// unwrap must then fail authentication.
	id := kekID(key2)
	copy(ct[offKeyID:], id[:])
	if _, err := k2.Open(ct, testAAD); !errors.Is(err, ErrDecrypt) {
		t.Errorf("Open with the wrong key: err = %v, want ErrDecrypt", err)
	}
}

func TestOpenTampered(t *testing.T) {
	k := newKeyring(t, newKey(t))
	ct := seal(t, k, []byte("a secret worth protecting"), testAAD)

	region := func(i int) (string, error) {
		switch {
		case i < offKeyID:
			return "version", ErrUnknownVersion
		case i < offWrapNonce:
			return "key ID", ErrUnknownKey
		case i < offWrappedDEK:
			return "wrap nonce", ErrDecrypt
		case i < offDataNonce:
			return "wrapped DEK", ErrDecrypt
		case i < offData:
			return "data nonce", ErrDecrypt
		case i < len(ct)-tagLen:
			return "data ciphertext", ErrDecrypt
		default:
			return "data tag", ErrDecrypt
		}
	}
	// Flip every bit of every byte.
	for i := range ct {
		name, want := region(i)
		for bit := range 8 {
			bad := bytes.Clone(ct)
			bad[i] ^= 1 << bit
			pt, err := k.Open(bad, testAAD)
			if !errors.Is(err, want) {
				t.Fatalf("%s: flipped bit %d of byte %d: Open = %q, %v; want %v", name, bit, i, pt, err, want)
			}
			if pt != nil {
				t.Fatalf("%s: Open returned plaintext alongside an error", name)
			}
		}
	}
}

func TestOpenUnknownVersion(t *testing.T) {
	k := newKeyring(t, newKey(t))
	ct := seal(t, k, []byte("value"), testAAD)
	for _, v := range []byte{0, 2, 0xff} {
		bad := bytes.Clone(ct)
		bad[0] = v
		if _, err := k.Open(bad, testAAD); !errors.Is(err, ErrUnknownVersion) {
			t.Errorf("version %d: err = %v, want ErrUnknownVersion", v, err)
		}
	}
	// The version is checked before the length.
	if _, err := k.Open([]byte{2}, testAAD); !errors.Is(err, ErrUnknownVersion) {
		t.Errorf("one byte, version 2: err = %v, want ErrUnknownVersion", err)
	}
}

func TestOpenUnknownKey(t *testing.T) {
	k := newKeyring(t, newKey(t))
	ct := seal(t, k, []byte("value"), testAAD)
	copy(ct[offKeyID:], []byte{0xde, 0xad, 0xbe, 0xef})
	_, err := k.Open(ct, testAAD)
	if !errors.Is(err, ErrUnknownKey) {
		t.Fatalf("err = %v, want ErrUnknownKey", err)
	}
	if !strings.Contains(err.Error(), "deadbeef") {
		t.Errorf("error %q does not name the key ID", err)
	}
}

func TestOpenTruncated(t *testing.T) {
	k := newKeyring(t, newKey(t))
	ct := seal(t, k, []byte("some value"), testAAD)
	for n := 0; n < len(ct); n++ {
		want := ErrDecrypt
		if n < Overhead {
			want = ErrMalformed
		}
		pt, err := k.Open(ct[:n], testAAD)
		if !errors.Is(err, want) || pt != nil {
			t.Errorf("truncated to %d bytes: Open = %q, %v; want %v", n, pt, err, want)
		}
	}
	if _, err := k.Open(nil, testAAD); !errors.Is(err, ErrMalformed) {
		t.Errorf("nil: err = %v, want ErrMalformed", err)
	}
	if _, err := k.Open(append(bytes.Clone(ct), 0), testAAD); !errors.Is(err, ErrDecrypt) {
		t.Errorf("extra byte: err = %v, want ErrDecrypt", err)
	}
}

func TestRotation(t *testing.T) {
	oldKey, newKeyBytes := newKey(t), newKey(t)
	before := newKeyring(t, oldKey)
	ct := seal(t, before, []byte("rotate me"), testAAD)

	after := newKeyring(t, newKeyBytes, oldKey)
	pt, err := after.Open(ct, testAAD)
	if err != nil || string(pt) != "rotate me" {
		t.Fatalf("new keyring Open of old ciphertext = %q, %v", pt, err)
	}

	// Re-sealing uses the new primary, which the old keyring can't open.
	resealed := seal(t, after, pt, testAAD)
	id := kekID(newKeyBytes)
	if !bytes.Equal(resealed[offKeyID:offWrapNonce], id[:]) {
		t.Error("re-sealed ciphertext does not use the new primary key")
	}
	if _, err := before.Open(resealed, testAAD); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("old keyring Open of new ciphertext: err = %v, want ErrUnknownKey", err)
	}
	// Once the old key is dropped, old ciphertexts are unreadable.
	if _, err := newKeyring(t, newKeyBytes).Open(ct, testAAD); !errors.Is(err, ErrUnknownKey) {
		t.Errorf("after dropping old key: err = %v, want ErrUnknownKey", err)
	}
}

func TestNewKeyringBadSizes(t *testing.T) {
	good := newKey(t)
	for _, n := range []int{0, 1, 16, 24, 31, 33, 64} {
		bad := bytes.Repeat([]byte{0xab}, n)
		_, err := NewKeyring(bad)
		if !errors.Is(err, ErrKeySize) {
			t.Errorf("primary of %d bytes: err = %v, want ErrKeySize", n, err)
		}
		_, err = NewKeyring(good, good, bad)
		if !errors.Is(err, ErrKeySize) {
			t.Errorf("old key of %d bytes: err = %v, want ErrKeySize", n, err)
		}
		if err != nil && n > 0 && strings.Contains(strings.ToLower(err.Error()), hex.EncodeToString(bad[:1])) {
			t.Errorf("error %q contains key material", err)
		}
	}
	if _, err := NewKeyring(nil); !errors.Is(err, ErrKeySize) {
		t.Errorf("nil primary: err = %v, want ErrKeySize", err)
	}
}

func TestNewKeyringCopiesKeys(t *testing.T) {
	key := newKey(t)
	orig := bytes.Clone(key)
	k := newKeyring(t, key)
	ct := seal(t, k, []byte("value"), testAAD)
	clear(key)
	if _, err := k.Open(ct, testAAD); err != nil {
		t.Fatalf("Open after caller cleared its key: %v", err)
	}
	if _, err := newKeyring(t, orig).Open(ct, testAAD); err != nil {
		t.Fatalf("Open with a keyring of the original key: %v", err)
	}
}

func TestNewKeyringDuplicateKey(t *testing.T) {
	key := newKey(t)
	k := newKeyring(t, key, key, bytes.Clone(key))
	ct := seal(t, k, []byte("value"), testAAD)
	if _, err := k.Open(ct, testAAD); err != nil {
		t.Fatalf("Open: %v", err)
	}
}

func TestErrorsDoNotLeak(t *testing.T) {
	key := newKey(t)
	k := newKeyring(t, key)
	plaintext := []byte("TOPSECRETVALUE")
	ct := seal(t, k, plaintext, testAAD)
	bad := bytes.Clone(ct)
	bad[len(bad)-1] ^= 1
	var errs []error
	for _, c := range [][]byte{nil, ct[:10], bad} {
		_, err := k.Open(c, testAAD)
		errs = append(errs, err)
	}
	_, err := k.Open(ct, []byte("wrong"))
	errs = append(errs, err)
	for _, err := range errs {
		if err == nil {
			t.Fatal("expected an error")
		}
		msg := err.Error()
		for _, s := range []string{string(plaintext), string(testAAD), hex.EncodeToString(key), string(key)} {
			if strings.Contains(msg, s) {
				t.Errorf("error %q leaks sensitive data", msg)
			}
		}
	}
}

func TestNilKeyring(t *testing.T) {
	var k *Keyring
	if _, err := k.Seal([]byte("x"), nil); err == nil {
		t.Error("Seal on nil Keyring succeeded")
	}
	if _, err := k.Open(make([]byte, Overhead), nil); err == nil {
		t.Error("Open on nil Keyring succeeded")
	}
}

func TestKeyringConcurrent(t *testing.T) {
	k := newKeyring(t, newKey(t), newKey(t))
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 50 {
				pt := []byte(strings.Repeat("v", g*50+i))
				ct, err := k.Seal(pt, testAAD)
				if err != nil {
					t.Error(err)
					return
				}
				got, err := k.Open(ct, testAAD)
				if err != nil || !bytes.Equal(got, pt) {
					t.Errorf("round trip failed: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
