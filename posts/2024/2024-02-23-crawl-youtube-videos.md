---
title: 'Hướng dẫn crawl Youtube Videos tự động với Python'
date: 2024-02-23
tags:
  - Data Collection
  - Python
---

Xin chào các bạn!

Trong bài post lần này mình sẽ giới thiệu một phương pháp để có thể tự động tải các videos/shorts/livestream từ Youtube về để có thêm nguồn data cho mô hình AI/ML.

Trong lĩnh vực AI/ML, data chiếm 80% công việc của các kĩ sư nên việc biết tận dụng các nền tảng trực tuyến để có thể kiếm thêm nhiều data sẽ rất có ích cho việc xây dựng một mô hình tốt. Có thể kể đến một vài nền tảng nổi tiếng để chúng ta scrape dữ liệu như Instagram, Tiktok, Reddit, Twitter (X), ...

Chúng ta bắt tay vào thực hành luôn nhé!

**Bước 1**: Install thư viện scrapetube, youtube-dl với pip
    
Cho bạn nào chưa biết, pip là một package manager của Python, nó giúp cho việc cài cắm đơn giản hơn rất nhiều so với việc ta tự download và install thủ công.

Package đầu tiên chúng ta sẽ cài là scrapetube, thư viện này sẽ giúp chúng ta **trích xuất urls** của các videos/shorts một cách tự động mà không cần phải mở một trình duyệt phụ như thư viện Selenium. Vì vậy, việc dùng thư viện này sẽ tiết kiệm khá nhiều thời gian và băng thông mạng của các bạn 😄. Package thứ hai và cũng là package cuối cùng là youtube_dl, thư viện này sẽ giúp chúng ta **tải về các Youtube urls** đã được trích xuất dùng.


Ta dùng đoạn lệnh sau để có thể cài 2 thư viện này nhé

```
pip install scrapetube youtube_dl
```
**Bước 2**: Code Python thôii

```python
# Step 1: Import những thư viện cần thiết
import scrapetube
import requests
import subprocess
import argparse
import pandas as pd

def download_video(url):
    # Đoạn này để setup chất lượng video bạn muốn tải, bạn có thể tham khảo kĩ hơn tại đường link mình để bên dưới nha.
    subprocess.call(["yt-dlp", "-f", "bestvideo[height<=1080][ext=mp4]", url])

channel_username = "schannelvn"  # Ví dụ về kênh youtube, các bạn có thể lấy kênh nào tùy thích nhé

# Có 3 options: videos, shorts, streams. Tùy các bạn muốn crawl dạng nào và bỏ vào content type nha, mặc định sẽ là videos.
urls  = scrapetube.get_channel(channel_username=channel_username, content_type="videos")


# Bắt đầu download thôiii
for url in urls:
    download_video(url)
```

**Kết luận**
Và thế  là chỉ sau 2 bước đơn giản, các bạn có thể crawl hàng tấn videos để có thêm data cho model rồi. Đơn giản phải không nào 😄

Phần tiếp theo, mình sẽ hướng dẫn các bạn crawl tiktok videos full HD, không watermarks nha, nhớ ghé blog tiếp hen !!


## Tải liệu tham khảo
1. [Scrapetube-demasmid][Scrapetube-demasmid]
2. [Youtube-dl][youtube-dl]


[Scrapetube-demasmid]: https://github.com/dermasmid/scrapetube
[youtube-dl]: https://github.com/ytdl-org/youtube-dl
