export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    // Handle OPTIONS request for CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    // Authentication token validation
    const authHeader = request.headers.get("Authorization");
    const expectedToken = env.AUTH_TOKEN || "upbit-portfolio-secret-key-2026";

    if (url.pathname === "/update") {
      if (request.method !== "POST") {
        return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
      }

      if (!authHeader || authHeader !== `Bearer ${expectedToken}`) {
        return new Response("Unauthorized", { status: 401, headers: corsHeaders });
      }

      try {
        const data = await request.json();
        // Save to KV namespace with key 'portfolio_data'
        await env.PORTFOLIO_KV.put("portfolio_data", JSON.stringify(data));
        return new Response("OK", { status: 200, headers: corsHeaders });
      } catch (err) {
        return new Response("Invalid JSON: " + err.message, { status: 400, headers: corsHeaders });
      }
    }

    if (url.pathname === "/data") {
      if (request.method !== "GET") {
        return new Response("Method Not Allowed", { status: 405, headers: corsHeaders });
      }

      const val = await env.PORTFOLIO_KV.get("portfolio_data");
      if (!val) {
        return new Response(JSON.stringify({ error: "No data found yet." }), {
          status: 404,
          headers: { "Content-Type": "application/json", ...corsHeaders }
        });
      }

      return new Response(val, {
        status: 200,
        headers: { "Content-Type": "application/json", ...corsHeaders }
      });
    }

    return new Response("Not Found", { status: 404, headers: corsHeaders });
  }
};
