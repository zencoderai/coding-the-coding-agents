from http.server import SimpleHTTPRequestHandler, HTTPServer
import os

PORT = 12345

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/regular":
            self.path = "regular.html"
        elif self.path == "/oops":
            self.path = "oops.html"
        elif self.path == "/notoops":
            self.path = "notoops.html"
        else:
            # fallback to 404 page if file not found
            self.send_error(404, "Page not found")
            return
        return super().do_GET()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # serve from current folder
    with HTTPServer(("", PORT), CustomHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()
