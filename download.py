import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
url_web = r"https://www.dlearningapp.com/web/DLFautoinsects.htm"
url_head=r"https://www.dlearningapp.com/web/"
ua = UserAgent()
headers = {"user-agent": ua.Chrome}
res = requests.get(url=url_web, headers=headers, verify=False)
soup = BeautifulSoup(res.text, "lxml")
tag_a = soup.find_all('a')
name = [i['href'] for i in tag_a]

download_url = [url_head+i for i in name]

file_names = []
for i in name:
    file_name = i.split("/")[1]
    file_names.append(file_name)
print(file_names)
for i, j in zip(download_url, file_names):
    with open("./images/"+j, 'wb+') as f:
        f.write(requests.get(i, headers=headers, verify=False).content)
