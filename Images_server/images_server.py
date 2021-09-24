from argparse import ArgumentParser
from collections import namedtuple
from contextlib import closing
from io import BytesIO
from json import dumps as json_encode
from json import load
from json.decoder import JSONDecodeError
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np

if sys.version_info >= (3, 0):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
    from urllib.parse import parse_qs
else:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn
    from urlparse import parse_qs

# Mapping the output format used in the client to the content type for the
# response
IMAGE_FORMATS = {"png": "image/png",
                 "jpeg": "image/jpeg",
                 "webp": "image/webp"}

ResponseStatus = namedtuple("HTTPStatus",
                            ["code", "message"])

HTTP_STATUS = {"OK": ResponseStatus(code = 200, message = "OK"),
               "Created" : ResponseStatus(code = 201, message = "Resource created"),
               "No_Content" : ResponseStatus(code = 204, message = "Resource modified"),
               "BAD_REQUEST" : ResponseStatus(code = 400, message = "Bad request"),
               "NOT_FOUND" : ResponseStatus(code = 404, message = "Not found"),
               "METHOD_NOT_ALLOWED" : ResponseStatus(code = 405, message = "Method not allowed"),
               "INTERNAL_SERVER_ERROR" : ResponseStatus(code = 500, message = "Internal server error"),
               "NOT_IMPLEMENTED": ResponseStatus(code = 501, message = "Method not implemented")}

ResponseData = namedtuple("ResponseData",
                          ["status", "content_type", "data_stream"])

CHUNK_SIZE = 1024
PROTOCOL = "http"
ROUTE_INDEX = "/"
ROUTE_IMAGE = "/image"
ROUTE_RESULT = "/result"
ROUTE_EXPORT = "/export"

test_img_arr = np.ones((28000, 28, 28), dtype = np.uint8) # global variable
predictions = np.ones((28000)) # global variable

class HTTPStatusError(Exception):
    """Exception wrapping a value from http.server.HTTPStatus"""

    def __init__(self, status, description = None):
        """
        Constructs an error instance from a tuple of
        (code, message, description), see http.server.HTTPStatus
        """
        super(HTTPStatusError, self).__init__()
        self.code = status.code
        self.message = status.message
        self.explain = description


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """An HTTP Server that handle each request in a new thread"""
    daemon_threads = True


class ChunkedHTTPRequestHandler(BaseHTTPRequestHandler):
    """"HTTP 1.1 Chunked encoding request handler"""
    # Use HTTP 1.1 as 1.0 doesn't support chunked encoding
    protocol_version = "HTTP/1.1"

    def query_get(self, queryData, key, default=""):
        """Helper for getting values from a pre-parsed query string"""
        return queryData.get(key, [default])[0]

    def do_GET(self):
        """Handles GET requests"""

        # Extract values from the query string
        path, _, query_string = self.path.partition('?')
        query = parse_qs(query_string)

        response = None
        
        print(u"[START]: Received GET for %s with query: %s" % (path, query))

        try:
            # Handle the possible request paths
            if path == ROUTE_INDEX:
                response = self.route_index(path, query)
            elif path == ROUTE_IMAGE:
                response = self.route_image(path, query)
            else:
                raise HTTPStatusError(HTTP_STATUS["NOT_FOUND"], "Resource not found")

            self.send_headers(response.status, response.content_type)
            self.stream_data(response.data_stream)

        except HTTPStatusError as err:
            # Respond with an error and log debug
            # information
            if sys.version_info >= (3, 0):
                self.send_error(err.code, err.message, err.explain)
            else:
                self.send_error(err.code, err.message)

            self.log_error(u"%s %s %s - [%d] %s", self.client_address[0],
                           self.command, self.path, err.code, err.explain)

        print("[END]")

    def route_index(self, path, query):
        """Handles routing for the application's entry point'"""

        try:
            response_body = str("Connected to server.")
            bytes_data = bytes(response_body, "utf-8") if sys.version_info >= (3, 0) \
                else bytes(response_body)
            return ResponseData(status = HTTP_STATUS["OK"], content_type = "text_html",
                                # Open a binary stream for reading the index
                                # HTML file
                                data_stream = BytesIO(bytes_data)
                                # if Python version is >= 3.6, this stream must be
                                # buffered (activated by default in binary mode of 
                                # Open() )
                                )

        except SystemError as err:
            # Couldn't open the stream
            raise HTTPStatusError(HTTP_STATUS["INTERNAL_SERVER_ERROR"],
                                  str(err))

    def route_image(self, path, query):
        """Handles routing for reading images"""

        global test_img_arr
        outputFormat = str("")

        # Get the parameters from the query string
        outputFormat = str(self.query_get(query, "outputFormat"))
        ImageID = self.query_get(query, "ImageID")

        # Validate the parameters, set error flag in case of unexpected
        # values
        if (len(str(ImageID)) == 0) or (int(ImageID) > 27999) or (int(ImageID) < 0)\
             or (outputFormat not in IMAGE_FORMATS):
            raise HTTPStatusError(HTTP_STATUS["BAD_REQUEST"],
                                  "Wrong parameters")
        else:
            try:
                print(u"[route_image]: Received GET for image_ID= %d & format= %s " \
                    % (int(ImageID), outputFormat ) )
                
                # Load an image
                img_array = np.array(np.reshape(test_img_arr[int(ImageID)], (28, 28)), 
                                    dtype = np.uint8)
                
                PIL_img = Image.fromarray(img_array, mode = "L")

                ImageStream = BytesIO()
                PIL_img.save(ImageStream, format = outputFormat)
                ImageStream.seek(0)

            except SystemError as err:
                # The service returned an error
                raise HTTPStatusError(HTTP_STATUS["INTERNAL_SERVER_ERROR"],
                                      str(err))
            else:
                return ResponseData(status = HTTP_STATUS["OK"],
                                    content_type = IMAGE_FORMATS[outputFormat],
                                    # Access the image buffered stream
                                    data_stream = ImageStream
                                    )
        
    def send_headers(self, status, content_type):
        """Send out the group of headers for a successful request"""

        # Send HTTP headers
        self.send_response(status.code, status.message)
        self.send_header('Content-type', content_type)
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Connection', 'close')
        self.end_headers()

    def stream_data(self, stream):
        """Consumes a stream in chunks to produce the response's output'"""
        
        print("Streaming started...")

        if stream:
            # Note: Closing the stream is important as the service throttles on
            # the number of parallel connections. Here we are using
            # contextlib.closing to ensure the close method of the stream object
            # will be called automatically at the end of the with statement's
            # scope.
            with closing(stream) as managed_stream:
                # Push out the stream's content in chunks
                while True:
                    data = managed_stream.read(CHUNK_SIZE)
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))

                    # If there's no more data to read, stop streaming
                    if not data:
                        break

                # Ensure any buffered output has been transmitted and close the
                # stream
                self.wfile.flush()

            print("Streaming completed.")

        else:
            # The stream passed in is empty
            self.wfile.write(b"0\r\n\r\n")
            print("Nothing to stream.")
    
    def do_POST(self):
        """Handles POST requests"""

        path = self.path

        status = None
        
        print(u"[START]: Received POST for %s" % (path))

        try:
            # Handle the possible POST requests
            if path.endswith('/'):
                status = HTTP_STATUS["METHOD_NOT_ALLOWED"]
            elif path == ROUTE_RESULT:
                status = self.route_result()
            elif path == ROUTE_EXPORT:
                status = self.route_export()
            else:
                raise HTTPStatusError(HTTP_STATUS["NOT_FOUND"], "Resource not found")

            self.send_response(int(status.code), str(status.message))
            self.end_headers()

        except HTTPStatusError as err:
            # Respond with an error and log debug
            # information
            if sys.version_info >= (3, 0):
                self.send_error(err.code, err.message, err.explain)
            else:
                self.send_error(err.code, err.message)

            self.log_error(u"%s %s %s - [%d] %s", self.client_address[0],
                           self.command, self.path, err.code, err.explain)

        print("[END]")

    def route_result(self):
        """Handles routing for saving results from inference"""
        
        global predictions

        try:
            length = int(self.headers['Content-Length'])
            with BytesIO() as f:
                f.write(self.rfile.read(length))
                f.seek(0)
                Json = load(f)
                print(u"[route_result]: Received POST with image_ID= %d & result= %d " \
                    % ( int(Json["ImageID"]), int(Json["result"]) ) )
                predictions[int(Json["ImageID"])] = int(Json["result"])

        except SystemError as err:
            # The service returned an error
            raise HTTPStatusError(HTTP_STATUS["INTERNAL_SERVER_ERROR"],
                                    str(err))
        else:
            return HTTP_STATUS["No_Content"]
    
    def route_export(self):
        """Handles routing for saving all the inference results in a csv file"""
        
        global predictions
        
        try:
            df_pred = pd.DataFrame({'ImageId' : np.arange(1, 28001, 1),
                            'Label' : predictions})
            df_pred.to_csv('output.csv', index = False)

        except SystemError as err:
            # The service returned an error
            raise HTTPStatusError(HTTP_STATUS["INTERNAL_SERVER_ERROR"],
                                    str(err))
        else:
            return HTTP_STATUS["No_Content"]


def load_images():
    global test_img_arr
    try:
        df_test = pd.read_csv("test.csv", sep=',')
        test_img_arr = np.reshape(np.array(df_test), (28000, 28, 28))

    except pd.errors.EmptyDataError as err:
        print("Cannot open/read csv test file")

def submit_results():
    # use kaggle API
    print()


# Define and parse the command line arguments
cli = ArgumentParser(description='Example Python Application')
cli.add_argument(
    "-p", "--port", type=int, metavar="PORT", dest="port", default=8000)
cli.add_argument(
    "--host", type=str, metavar="HOST", dest="host", default="localhost")
arguments = cli.parse_args()

# If the module is invoked directly, initialize the application
if __name__ == '__main__':
    # Create and configure the HTTP server instance
    server = ThreadedHTTPServer((arguments.host, arguments.port),
                                ChunkedHTTPRequestHandler)
    print("Starting server, use <Ctrl-C> to stop...")
    print(u"Open {0}://{1}:{2}{3} in a web browser.".format(PROTOCOL,
                                                            arguments.host,
                                                            arguments.port,
                                                            ROUTE_INDEX))

    load_images()
    
    try:
        # Listen for requests indefinitely
        server.serve_forever()

    except KeyboardInterrupt:
        # A request to terminate has been received, stop the server
        print("\nShutting down...")
        server.socket.close()
