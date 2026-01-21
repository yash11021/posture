# PostSURE Frontend

TypeScript + MediaPipe real-time posture detection.

## Development

```bash
npm install
npm run dev
```

## Environment Variables

Create `.env.local` for local development:
```
VITE_API_URL=http://localhost:7860
```

For production (Vercel), set:
```
VITE_API_URL=https://your-hf-space.hf.space
```

## Build

```bash
npm run build
```

Output in `dist/` for deployment.
