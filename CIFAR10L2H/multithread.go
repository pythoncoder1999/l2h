// multithread.go (final)
package main

import (
	"encoding/binary"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

const (
	pythonServerAddr     = "localhost:6000"
	goServerPort         = ":7000"
	maxConcurrentClients = 1000
	clientTimeout        = 30 * time.Second
)

var (
	connLimiter = make(chan struct{}, maxConcurrentClients)
	wg          sync.WaitGroup
)

func setDeadline(c net.Conn) { _ = c.SetDeadline(time.Now().Add(clientTimeout)) }
func readFull(c net.Conn, buf []byte) error { setDeadline(c); _, err := io.ReadFull(c, buf); return err }
func writeAll(c net.Conn, buf []byte) error { setDeadline(c); _, err := c.Write(buf); return err }

func main() {
	ln, err := net.Listen("tcp", goServerPort)
	if err != nil {
		log.Fatalf("[Go Proxy] Failed to start on %s: %v", goServerPort, err)
	}
	defer ln.Close()
	log.Printf("[Go Proxy] Listening on %s -> %s", goServerPort, pythonServerAddr)

	go gracefulShutdown(ln)

	for {
		clientConn, err := ln.Accept()
		if err != nil {
			log.Printf("[Go Proxy] Accept error: %v", err)
			continue
		}
		if tcp, ok := clientConn.(*net.TCPConn); ok {
			_ = tcp.SetKeepAlive(true)
			_ = tcp.SetKeepAlivePeriod(30 * time.Second)
		}
		connLimiter <- struct{}{}
		wg.Add(1)
		go func(conn net.Conn) {
			defer func() { <-connLimiter; wg.Done() }()
			handleClient(conn)
		}(clientConn)
	}
}

func gracefulShutdown(ln net.Listener) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	<-c
	log.Println("[Go Proxy] Shutting down...")
	_ = ln.Close()
	wg.Wait()
	log.Println("[Go Proxy] Bye.")
	os.Exit(0)
}

func handleClient(clientConn net.Conn) {
	defer clientConn.Close()

	// 1) Command
	cmd := make([]byte, 1)
	if err := readFull(clientConn, cmd); err != nil { log.Printf("[Go Proxy] Read cmd: %v", err); return }
	if cmd[0] != 'B' { log.Printf("[Go Proxy] Unknown cmd: %v", cmd[0]); return }

	// 2) Lengths
	lengthBuf := make([]byte, 8)
	if err := readFull(clientConn, lengthBuf); err != nil { log.Printf("[Go Proxy] Read lengths: %v", err); return }
	batchLen := binary.BigEndian.Uint32(lengthBuf[0:4])
	rejLen := binary.BigEndian.Uint32(lengthBuf[4:8])

	// 3) Payload
	totalLen := int(batchLen) + int(rejLen)
	payload := make([]byte, totalLen)
	if err := readFull(clientConn, payload); err != nil {
		log.Printf("[Go Proxy] Read payload %dB: %v", totalLen, err)
		return
	}

	// 4) Connect Python expert
	pyConn, err := net.DialTimeout("tcp", pythonServerAddr, clientTimeout)
	if err != nil { log.Printf("[Go Proxy] Connect Python %s: %v", pythonServerAddr, err); return }
	defer pyConn.Close()
	if ptcp, ok := pyConn.(*net.TCPConn); ok {
		_ = ptcp.SetKeepAlive(true)
		_ = ptcp.SetKeepAlivePeriod(30 * time.Second)
	}

	// 5) Client IP
	clientIP := "unknown"
	if ra, ok := clientConn.RemoteAddr().(*net.TCPAddr); ok && ra.IP != nil { clientIP = ra.IP.String() }
	ipBytes := []byte(clientIP)
	ipLenBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(ipLenBuf, uint32(len(ipBytes)))

	// 6) Forward to Python
	if err := writeAll(pyConn, ipLenBuf); err != nil { log.Printf("[Go Proxy] Send ipLen: %v", err); return }
	if err := writeAll(pyConn, ipBytes); err != nil { log.Printf("[Go Proxy] Send ip: %v", err); return }
	if err := writeAll(pyConn, cmd); err != nil { log.Printf("[Go Proxy] Send cmd: %v", err); return }
	if err := writeAll(pyConn, lengthBuf); err != nil { log.Printf("[Go Proxy] Send lengths: %v", err); return }
	if err := writeAll(pyConn, payload); err != nil { log.Printf("[Go Proxy] Send payload: %v", err); return }

	// 7) Read Python response
	respLenBuf := make([]byte, 4)
	if err := readFull(pyConn, respLenBuf); err != nil { log.Printf("[Go Proxy] Read respLen: %v", err); return }
	respLen := binary.BigEndian.Uint32(respLenBuf)
	resp := make([]byte, respLen)
	if err := readFull(pyConn, resp); err != nil { log.Printf("[Go Proxy] Read resp %dB: %v", respLen, err); return }

	// 8) Relay back to client
	if err := writeAll(clientConn, respLenBuf); err != nil { log.Printf("[Go Proxy] Write respLen: %v", err); return }
	if err := writeAll(clientConn, resp); err != nil { log.Printf("[Go Proxy] Write resp: %v", err); return }
}
