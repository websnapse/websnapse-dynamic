# WebSnapse Backend

### Prerequisites

- Python 3.9.6 or higher
- Pydantic
- Numpy
- FastAPI
- Uvicorn

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/websnapse/BackEnd.git
   ```
2. Install Python packages
   ```sh
   pip install -r requirements.txt
   ```
3. Run the server
   ```sh
   uvicorn main:app --reload
   ```
4. Go to http://localhost:8000/docs/default/ to see the API documentation.

5. Use Postman or any other API client to send requests to the API.

### Routes

- `POST` /simulate/
- `POST` /simulate/last/
- `POST` /simulate/step/

### Testing

To test, run the following command:

```sh
python tests.py
```
