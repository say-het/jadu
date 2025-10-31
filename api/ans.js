export default function handler(req, res) {
  res.send(`
   curl -X POST "https://jadu-seven.vercel.app/api/gemini" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain black holes in simple words."}'

`);
}
