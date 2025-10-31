export default function handler(req, res) {
  res.send(`
    Invoke-RestMethod -Uri "https://jadu-seven.vercel.app/api/gemini"   -Method POST   -ContentType "application/json"  -Body '{"prompt": "Explain black holes in simple words."}' | ConvertTo-Json

`);
}
