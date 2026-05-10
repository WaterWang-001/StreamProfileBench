"""Prompt builder for StreamProfileBench.

Used by ``bench_inference.py`` to (re)assemble the per-step user prompt at
inference time, threading the ``persona_summary`` from the previous step into
the next step's prompt. Candidate pools and ground-truth labels are pre-built
into the data files; this module only formats them for the model.
"""

import json


class BenchTaskBuilder:
    PLATFORM_CONTEXT = {
        "weibo": {
            "name": "微博（Weibo）",
            "desc": "中国微博客平台。用户通过发帖、转发、评论参与话题讨论。",
            "tag_meaning": "tag 是用户发帖/转发时使用的 #话题标签#，反映用户当前关注的热点、追星、生活话题。",
            "hint": "注意区分：用户的长期兴趣（如固定追的明星）vs 临时热点（如突发新闻）。转发行为往往比原创发帖更能体现真实兴趣。",
        },
        "xiaohongshu": {
            "name": "小红书（Xiaohongshu）",
            "desc": "生活方式分享平台。用户发布图文/视频笔记，内容涵盖美妆、穿搭、美食、旅行、母婴等。",
            "tag_meaning": "tag 是笔记中标注的话题标签，反映用户的内容创作方向和生活兴趣。",
            "hint": "小红书用户的兴趣通常围绕具体的生活场景。关注笔记标题中的关键词，它们往往比正文更能概括主题。",
        },
        "toutiao": {
            "name": "今日头条（Toutiao）",
            "desc": "资讯与短视频平台。用户以浏览和创作短视频/图文为主。",
            "tag_meaning": "tag 是内容创作者在标题中标注的话题标签，反映用户的内容创作领域。",
            "hint": "头条用户的内容多为短视频，标题是最重要的信号。注意用户是否有固定的创作领域（如美食、旅游、育儿）。",
        },
        "zhihu": {
            "name": "知乎（Zhihu）",
            "desc": "问答与长文社区。用户通过提问、回答、写文章参与知识分享。",
            "tag_meaning": "tag 是用户浏览、回答或发布的问题标题/话题，反映用户的知识兴趣和专业领域。",
            "hint": "知乎 tag 通常是完整的问题标题（较长）。关注用户回答的领域集中度——专业用户往往在 2-3 个领域深耕。",
        },
        "douban": {
            "name": "豆瓣（Douban）",
            "desc": "影视/图书/音乐评论社区。用户通过标记作品（想看/看过/在读等）记录文化消费。",
            "tag_meaning": "tag 是用户对作品的标记行为，格式为 '动作:作品名'（如 '看过影视:片名'、'想读图书:书名'），反映用户的文化消费偏好。",
            "hint": "豆瓣用户的兴趣体现在作品类型（电影/书/音乐）和题材偏好上。注意 '想看' vs '看过' 的区别——前者是意向，后者是已消费。",
        },
    }

    def __init__(self, platform):
        self.platform = platform

    def format_prompt(self, username, bio, batch_n_posts, candidate_pool,
                      step_id, total_steps, prev_persona):
        pool_str = json.dumps(candidate_pool, ensure_ascii=False)
        pool_size = len(candidate_pool)
        n_to_select = max(1, round(pool_size * 0.25))
        ctx = self.PLATFORM_CONTEXT.get(self.platform, {})

        persona_block = (
            f"## 当前用户画像（基于历史观测积累）\n{prev_persona}"
            if prev_persona else
            "## 当前用户画像\n暂无历史观测，这是该用户的第一个活动批次。请从零开始构建画像。"
        )

        prompt = f"""# 任务：流式用户画像维护与兴趣预测

你是一个用户画像系统，负责处理**流式社交媒体数据**。每收到一批新的用户活动数据，你需要：
1. **更新画像**：基于新活动和已有画像，维护对该用户兴趣、偏好和行为模式的全面理解
2. **预测兴趣**：从候选池中选出该用户在下一个活动周期最可能参与的 tag

## 平台背景
**{ctx.get('name', self.platform)}** — {ctx.get('desc', '')}
- **Tag 含义**：{ctx.get('tag_meaning', '')}
- **分析提示**：{ctx.get('hint', '')}

## 用户基本信息
- **用户名**：{username}
- **个人简介**：{bio if bio else "未提供"}

{persona_block}

## 新活动数据（第 {step_id} 批）
{batch_n_posts}

## 候选标签池（共 {pool_size} 个）
从以下候选池中，**选出恰好 {n_to_select} 个**该用户在下一个活动周期最可能参与的 tag。
注意：
- 你必须从候选池中选出恰好 {n_to_select} 个标签，不多不少
- 你的目标是预测用户**未来**的行为，而非总结用户过去的行为
- 用户的兴趣会随时间演变：有些当前话题只是一时热度，下个周期可能不再参与；
    有些当前未出现的话题，可能因用户的潜在偏好而在未来浮现


{pool_str}

## 输出格式（严格 JSON）
{{
  "persona_summary": "更新后的用户画像摘要：包含用户的核心兴趣领域、行为模式、偏好特征等（将传递给下一批次）",
  "predicted_tags": ["tag1", "tag2", ...],
  "reasoning": "简要说明预测依据：画像中的哪些特征支持了你的标签选择"
}}
"""
        return prompt
