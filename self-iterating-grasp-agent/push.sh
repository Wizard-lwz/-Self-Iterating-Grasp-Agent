#!/bin/bash
# 一键推送脚本 - 请先在 GitHub 创建空仓库后运行

set -e

echo "=================================="
echo "Self-Iterating Grasp Agent"
echo "一键推送到 GitHub"
echo "=================================="
echo ""

# 检查是否已配置 git 身份
if ! git config user.name > /dev/null 2>&1; then
    echo "⚠️  请先配置 git 身份信息:"
    echo "   git config user.name \"你的GitHub用户名\""
    echo "   git config user.email \"你的GitHub邮箱\""
    exit 1
fi

echo "✅ Git 身份: $(git config user.name) <$(git config user.email)>"
echo ""

# 提示用户输入仓库信息
read -p "请输入你的 GitHub 用户名: " username
read -p "使用 SSH 还是 HTTPS? (ssh/https，默认 ssh): " protocol
protocol=${protocol:-ssh}

if [ "$protocol" = "ssh" ]; then
    remote_url="git@github.com:$username/self-iterating-grasp-agent.git"
else
    remote_url="https://github.com/$username/self-iterating-grasp-agent.git"
fi

echo ""
echo "将推送到: $remote_url"
read -p "确认继续? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

# 更新 commit 作者
echo ""
echo "📝 更新 commit 作者信息..."
git commit --amend --reset-author --no-edit

# 添加 remote
if git remote | grep -q origin; then
    echo "⚠️  remote 'origin' 已存在，删除旧的..."
    git remote remove origin
fi
git remote add origin "$remote_url"

# 推送
echo ""
echo "🚀 推送到 GitHub..."
git push -u origin main

echo ""
echo "=================================="
echo "✅ 推送成功！"
echo "=================================="
echo ""
echo "访问你的仓库:"
echo "👉 https://github.com/$username/self-iterating-grasp-agent"
echo ""
echo "建议后续操作:"
echo "1. 在 GitHub 仓库页面添加 Topics: robotics, reinforcement-learning, model-based-rl, manipulation"
echo "2. 在 About 填写描述: Self-iterating grasp agent, 25%→100% solve rate"
echo "3. 检查 README 显示效果（GIF 和曲线图是否正常）"
echo ""
