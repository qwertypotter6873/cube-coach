export default {
  base: "./",
  server: { host: "127.0.0.1", port: 5180 },
  build: { rollupOptions: { input: { main: "index.html", voices: "voices.html" } } },
};
