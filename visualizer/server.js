const fs = require("fs");
const http = require("http");
const path = require("path");

const host = "0.0.0.0";
const port = 8080;

const contentTypeByExt = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
};

const server = http.createServer((req, res) => {
    const requestPath = req.url === "/" ? "/index.html" : req.url;
    const filePath = path.join(__dirname, decodeURIComponent(requestPath));

    fs.readFile(filePath, (error, data) => {
        if (error) {
            res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
            res.end("Not found");
            return;
        }

        const extension = path.extname(filePath);
        const contentType = contentTypeByExt[extension] || "application/octet-stream";
        res.writeHead(200, { "Content-Type": contentType });
        res.end(data);
    });
});

server.listen(port, host, () => {
    console.log(`Visualizer running at http://${host}:${port}`);
});
