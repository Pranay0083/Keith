package memstore_test

import (
	"github.com/Pranay0083/keith/internal/worker"
	"github.com/Pranay0083/keith/internal/worker/memstore"
)

var _ worker.Store = (*memstore.Store)(nil)
