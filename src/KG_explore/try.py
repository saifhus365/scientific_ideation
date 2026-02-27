from requests import get

API_CALL = "https://opencitations.net/index/api/v2/references/doi:10.1101/2025.04.23.650337"
HTTP_HEADERS = {"authorization": "fabb9c8a-c2d8-4018-9c2f-4c091634946c "}

response = get(API_CALL, headers=HTTP_HEADERS)

print(response.text)