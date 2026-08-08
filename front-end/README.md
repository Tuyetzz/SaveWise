# DisasterResponse — voice triage console

Single-page front-end for the Phase 1 backend: microphone in with client-side
endpointing (energy VAD or push-to-talk), question audio out (24 kHz PCM over
WebSocket), live transcript feed, and the six clinical fields with
known/unknown state.

## Run

```bash
bun run dev
```

Note: `bun run build` segfaults with Bun 1.3.13 on this machine (a Bun bug).
Node works: `node node_modules/next/dist/bin/next dev` (or `build`).

Open http://localhost:3000, check the server URL in the input (defaults to
`ws://<page-host>:8000/ws/converse`), press **Start interview** and allow the
microphone.

## Notes

- Microphone requires a secure context: `http://localhost` is fine on desktop;
  a phone needs the page over HTTPS **and** the backend over `wss://` (mkcert
  cert on the backend, accept it on the phone at `https://<ip>:8000/health`).
- Auto voice mode opens a segment after ~100 ms above the energy threshold and
  closes it after ~900 ms of silence; switch to **Push to talk** in noisy rooms.
- The mic is gated while a question is playing (half-duplex, matching the
  backend's strictly sequential turn loop).
