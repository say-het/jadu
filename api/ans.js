export default function handler(req, res) {
  res.send(`
   Invoke-RestMethod -Uri "https://jadu-seven.vercel.app/" -Method POST -ContentType "application/json" -Body '{"prompt": "Explain black holes in simple words."}'
`);
}
