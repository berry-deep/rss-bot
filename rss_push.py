#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSS 每日聚合推送到微信（PushPlus）
- feedparser 抓取 RSS
- requests 抓取每篇文章正文
- OpenAI 兼容 API 翻译与总结
- PushPlus API 推送
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import feedparser
import requests
import re
import html
import time
import random
from datetime import datetime, timedelta, timezone
from openai import OpenAI

# ── 配置项 ──────────────────────────────────────────────────────────
RSS_URL      = "https://marginalrevolution.com/feed"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
# 优先从环境变量读取（GitHub Actions Secrets），无则用默认值
PUSH_TOKEN   = __import__("os").environ.get("PUSHPLUS_TOKEN",    "1cb3a19ad4fc454c8d10689f3bd2b551")
PUSH_TITLE   = "今日 RSS 聚合"
TIME_WINDOW  = 24   # 小时
TIMEOUT      = 15   # 文章抓取超时（秒）
MAX_ARTICLES = 20   # 最多抓取文章数，防止跑太久
SLEEP_RANGE  = (1, 3)   # 两次请求之间随机休眠秒数

# ── AI 大模型配置（OpenAI 兼容格式；当前走 OpenRouter）───────────────
API_KEY    = __import__("os").environ.get("OPENAI_API_KEY",    "")
BASE_URL   = __import__("os").environ.get("OPENAI_BASE_URL",   "https://openrouter.ai/api/v1")
MODEL_NAME = __import__("os").environ.get("OPENAI_MODEL",      "deepseek/deepseek-chat-v3.1:free")
AI_TIMEOUT = 120   # 大模型请求超时（秒）

SYSTEM_PROMPT = (
    "你是一个专业的经济与科技专栏翻译官。"
    "请将用户发来的英文 RSS 内容转化为排版清晰的中文简报。"
    "要求：每篇文章输出【原英文标题】、【中文翻译标题】、【原文链接】，"
    "并用无序列表提炼 3-5 个中文核心观点（摒弃废话，保留硬核数据和逻辑）。"
    "最终输出请使用 Markdown 格式。"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

# ── 工具函数 ─────────────────────────────────────────────────────────

def strip_html(raw: str) -> str:
    """清理 HTML：删 script/style → 去标签 → 解码实体 → 折叠空白"""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def within_24h(published_parsed) -> bool:
    """判断文章的发布时间是否在过去 24 小时内"""
    if not published_parsed:
        return False
    pub_dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - pub_dt) <= timedelta(hours=TIME_WINDOW)


def polite_sleep():
    """两次请求之间随机休眠，降低对目标站点的压力"""
    t = random.uniform(*SLEEP_RANGE)
    time.sleep(t)


def _find_div_bounds(raw: str, marker: str) -> str | None:
    """
    在 raw HTML 中找到 <div class="marker"> 对应的闭合 </div> 之间的内容。
    逐字符追踪嵌套深度，确保在多级嵌套时也能正确截断。
    返回截取到的 inner HTML 字符串（不含外层 div），找不到返回 None。
    """
    idx = raw.find(marker)
    if idx < 0:
        return None
    tag_start = raw.rfind('<div', 0, idx)
    if tag_start < 0:
        return None
    start_content = raw.find('>', tag_start) + 1
    if start_content <= tag_start:
        return None
    depth = 1
    i = start_content
    n = len(raw)
    while i < n and depth > 0:
        if raw[i] == '<':
            nxt = raw[i+1:i+5].lower()
            if nxt == 'div ' or nxt == 'div>':
                depth += 1
                i += 1
            elif nxt == '/div':
                depth -= 1
                i += 1
                if depth == 0:
                    return raw[start_content:i]
        i += 1
    return raw[start_content:i]


def extract_full_text(url: str) -> str:
    """访问文章页面，提取正文纯文本"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.text

        inner = _find_div_bounds(raw, 'class="entry-content"')
        if inner is None:
            inner = _find_div_bounds(raw, "class='entry-content'")
        if inner is not None:
            return strip_html(inner)

        inner = _find_div_bounds(raw, 'class="post-content"')
        if inner is not None:
            return strip_html(inner)

        m = re.search(r"(?is)<body[^>]*>(.*?)</body>", raw, re.DOTALL)
        if m:
            return strip_html(m.group(1))

        return strip_html(raw)

    except Exception as e:
        print(f"  [!] 正文抓取失败 [{url}]: {e}")
        return ""


# ── AI 翻译与总结 ─────────────────────────────────────────────────────

def ai_translate(articles: list[dict]) -> str:
    """
    将合并后的英文文章全文发给大模型，返回翻译后的中文 Markdown 简报。
    异常时抛出 Exception，由上层捕获并推送错误通知。
    """
    if not articles:
        return "今日暂无新文章。"

    # 组装待翻译的纯文本
    sections = []
    for i, art in enumerate(articles, 1):
        body = art.get("full_text") or strip_html(art.get("summary", ""))
        sections.append(
            f"[文章 {i}]\n"
            f"原标题：{art['title']}\n"
            f"链接：{art['link']}\n"
            f"正文：\n{body[:5000]}\n"
        )

    user_content = "\n\n".join(sections)

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=AI_TIMEOUT,
        default_headers={
            "HTTP-Referer": "https://github.com/berry-deep/rss-bot",
            "X-Title":      "RSS Daily Push",
        },
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


# ── 主流程 ───────────────────────────────────────────────────────────

def fetch_rss(url: str) -> list[dict]:
    """抓取 RSS，筛选近 24 小时文章"""
    print("[*] 正在抓取 RSS feed ...")
    feed = feedparser.parse(url)
    if feed.bozo:
        print(f"[!] RSS 解析警告: {feed.bozo_exception}")

    articles = []
    for entry in feed.entries:
        if within_24h(getattr(entry, "published_parsed", None)):
            articles.append({
                "title":   entry.get("title", "无标题"),
                "link":    entry.get("link",  ""),
                "summary": entry.get("summary", ""),
            })
        if len(articles) >= MAX_ARTICLES:
            break

    print(f"[*] RSS 中近 24 小时文章：{len(articles)} 篇")
    return articles


def fetch_full_text(articles: list[dict]) -> list[dict]:
    """逐篇抓取正文"""
    print("[*] 正在抓取文章正文 ...")
    for i, art in enumerate(articles, 1):
        print(f"  [{i}/{len(articles)}] {art['title'][:60]} ...", end="", flush=True)
        art["full_text"] = extract_full_text(art["link"])
        if art["full_text"]:
            print(" [OK]")
        else:
            print(" [FAIL - use summary]")
        polite_sleep()
    return articles


def build_raw_message(articles: list[dict]) -> str:
    """将英文原文拼接为 Markdown（AI 调用失败时的降级推送内容）"""
    if not articles:
        return "[-] 过去 24 小时暂无新文章。"

    lines = [
        f"## {PUSH_TITLE}（原文，AI 翻译失败）",
        f"> 抓取时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"> 文章数量：{len(articles)}",
        "",
    ]
    for i, art in enumerate(articles, 1):
        lines.append(f"### {i}. {art['title']}")
        lines.append(f"[原文链接]({art['link']})")
        lines.append("")
        body = art.get("full_text") or strip_html(art.get("summary", ""))
        lines.append(body[:3000])
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def push_to_wechat(token: str, title: str, content: str) -> dict:
    """发送 POST 请求到 PushPlus API"""
    payload = {
        "token":    token,
        "title":    title,
        "content":  content,
        "template": "markdown",
    }
    resp = requests.post(PUSHPLUS_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── 入口 ─────────────────────────────────────────────────────────────

def main():
    articles = fetch_rss(RSS_URL)

    if not articles:
        print("[!] 没有找到符合条件的文章，跳过推送。")
        return

    fetch_full_text(articles)

    # ── AI 翻译与总结 ────────────────────────────────────────────────
    print("[*] 正在调用大模型翻译与总结 ...")
    try:
        chinese_md = ai_translate(articles)
        print("[*] AI 翻译完成，正在推送到微信 ...")
        result = push_to_wechat(PUSH_TOKEN, PUSH_TITLE, chinese_md)
        if result.get("code") == 200:
            print("[OK] 推送成功！")
        else:
            print(f"[ERR] 推送失败：{result}")
    except Exception as e:
        error_msg = (
            f"## {PUSH_TITLE}（AI 翻译失败）\n\n"
            f"> 抓取时间：{datetime.now():%Y-%m-%d %H:%M}\n\n"
            f"**AI 翻译与总结环节出错：**\n\n"
            f"```\n{e}\n```\n\n"
            f"---\n\n"
            f"以下是原文（未翻译）：\n\n"
        )
        raw_md = build_raw_message(articles)
        fallback = error_msg + raw_md
        print(f"[ERR] AI 调用失败：{e}，推送原文降级内容 ...")
        result = push_to_wechat(PUSH_TOKEN, f"{PUSH_TITLE}（AI 失败-降级原文）", fallback)
        if result.get("code") == 200:
            print("[OK] 降级推送成功！")
        else:
            print(f"[ERR] 降级推送也失败：{result}")


if __name__ == "__main__":
    main()
sync rss_push.py: switch to OpenRouter
