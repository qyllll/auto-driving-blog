# Elon's 自动驾驶技术博客

基于 **Hugo + PaperMod** 的个人技术博客，专注 VLA / 世界模型 / 端到端自动驾驶知识分享。

---

## 🚀 5 分钟上线（Vercel 免费部署）

### 第 1 步：把项目推到 GitHub

```bash
# 在项目目录下
git add .
git commit -m "初始化博客"
```

然后在 GitHub 上新建一个仓库（比如叫 `auto-driving-blog`），按 GitHub 给的提示把代码推上去：

```bash
git remote add origin https://github.com/你的用户名/auto-driving-blog.git
git branch -M main
git push -u origin main
```

### 第 2 步：连接 Vercel（一次配置，永久自动部署）

1. 打开 [vercel.com](https://vercel.com)，用 GitHub 账号登录
2. 点 **"Add New Project"** → 选刚才的 `auto-driving-blog` 仓库
3. Vercel 会**自动识别**这是 Hugo 项目（不用改任何设置）
4. 点 **"Deploy"**

几十秒后你的博客就上线了，地址形如 `auto-driving-blog.vercel.app`。

> ✅ **以后每次 `git push`，Vercel 自动重新构建并上线，你永远不用手动点发布。**

### 第 3 步（可选）：绑定个人域名

在 Vercel 项目的 Settings → Domains 里添加你的域名。Vercel 会自动配 HTTPS。

---

## ✍️ 写新文章的工作流（日常只用做这个）

### 方法一：复制模板（推荐，最简单）

1. 复制一个现成文章，比如：
   ```bash
   cp content/posts/knowledge/什么是VLA模型.md content/posts/knowledge/我的新文章.md
   ```
2. 改 `title`、`tags`、`summary` 和正文
3. 把 `draft: true` 改成 `draft: false`（否则不会发布）
4. `git push` → 自动上线 ✅

### 方法二：直接在 GitHub 网页上写

在 GitHub 仓库页面点 `content/posts/knowledge/` → "Add file" → 直接写 Markdown → Commit。
**连本地都不用开**，手机都能发文章。

### 文章放哪个文件夹？

| 文件夹 | 用途 | 示例 |
|--------|------|------|
| `content/posts/knowledge/` | 知识点拆解 | 什么是VLA、什么是世界模型 |
| `content/posts/paper-reading/` | 论文精读 | DriveVLM精读 |
| `content/posts/weekly/` | 每周速递 | 本周论文速递 |
| `content/posts/projects/` | 项目实践 | （后续补充） |
| `content/posts/interview/` | 求职面经 | （后续补充） |

### 文章头部怎么填（frontmatter）

```yaml
---
title: "文章标题"
date: 2026-07-17
draft: false              # ← false 才会发布！true 是草稿
categories: ["知识点拆解"]  # 栏目分类
tags: ["VLA", "世界模型"]   # 标签，随便加
summary: "一句话简介，显示在列表里"
---
```

---

## 🤖 让 AI 帮你写内容

你可以让我（或任何 AI）帮你生成文章。只要：

1. 告诉我主题（比如"写一篇关于 Flow Matching 在自动驾驶里应用的知识点拆解"）
2. 我按本博客的格式生成 Markdown
3. 你审阅修改后放进 `content/posts/对应栏目/`
4. `git push` 上线

**最适合 AI 生成的栏目**：知识点拆解、论文精读、每周速递
**建议你自己写的栏目**：项目实践、求职面经（这些有你的一手经历才真实）

---

## 🎨 怎么改网站外观

所有配置在 `hugo.yaml`：

| 想改什么 | 改哪里 |
|----------|--------|
| 网站名字 | `title` 和 `params.label.text` |
| 首页自我介绍 | `params.homeInfoParams.Content` |
| 社交链接 | `params.socialIcons`（改 url 成你的） |
| 默认深色/浅色 | `params.defaultTheme: dark/light/auto` |
| 导航菜单 | `menu.main` |

改完 `git push` 即生效。

---

## 📂 项目结构

```
auto-driving-blog/
├── hugo.yaml                  # 站点配置（改这里调外观）
├── vercel.json                # Vercel 部署配置
├── content/
│   ├── about.md               # 关于页
│   ├── search.md              # 搜索页
│   ├── archives.md            # 归档页
│   └── posts/                 # 所有博客文章
│       ├── knowledge/         # 💡 知识点拆解
│       ├── paper-reading/     # 📄 论文精读
│       └── weekly/            # 📰 每周速递
├── archetypes/                # 新文章模板
├── static/                    # 图片等静态资源
└── themes/PaperMod/           # 主题（不用动）
```

---

## ❓ 常见问题

**Q: 本地预览一定要装 Hugo 吗？**
A: 不用。直接 push 到 GitHub，Vercel 自动构建。想本地预览的话，装 Hugo Extended v0.146+ 然后运行 `hugo server -D`。

**Q: Vercel 真的免费吗？**
A: 个人用完全免费（Hobby 计划），流量和部署次数都够用。

**Q: 国内访问慢怎么办？**
A: 可以换 Cloudflare Pages（同样免费，国内 CDN 更好），部署流程几乎一样。

**Q: 怎么加文章配图？**
A: 图片放 `static/images/`，文章里用 `![描述](/images/xxx.png)` 引用。

---

## 📝 当前进度

- [x] 站点框架 + PaperMod 主题
- [x] 知识点拆解 × 3（VLA / 世界模型 / 端到端）
- [x] 论文精读 × 1（DriveVLM）
- [x] 内容模板（论文精读 / 知识点 / 周报）
- [x] Vercel 自动部署配置
- [ ] 关于页完善（等你准备好简历信息）
- [ ] 项目实践栏目（等你有了开源项目）
- [ ] 绑定个人域名
