# 🚀 GitHub 推送指南

## 前置准备

1. **在 GitHub 上创建空仓库**
   - 访问 https://github.com/new
   - 仓库名: `self-iterating-grasp-agent`（或你喜欢的名字）
   - **不要勾选** "Add a README file"（我们已经有了）
   - 选择 Public 或 Private
   - 点击 "Create repository"

2. **更新 git 作者信息**（替换占位符）
   ```bash
   cd ~/self-iterating-grasp-agent
   git config user.name "你的GitHub用户名"
   git config user.email "你的GitHub邮箱"
   git commit --amend --reset-author --no-edit   # 更新最后一个commit的作者
   ```

## 推送命令

### 使用 SSH（推荐，需要先配置 SSH key）
```bash
cd ~/self-iterating-grasp-agent
git remote add origin git@github.com:你的用户名/self-iterating-grasp-agent.git
git push -u origin main
```

### 使用 HTTPS（会要求输入 GitHub 用户名和 token）
```bash
cd ~/self-iterating-grasp-agent
git remote add origin https://github.com/你的用户名/self-iterating-grasp-agent.git
git push -u origin main
```

## 推送后

访问 `https://github.com/你的用户名/self-iterating-grasp-agent`，你会看到：
- ✅ README 自动渲染，带 GIF 和曲线图
- ✅ 四版实验日志在 `outputs/` 目录
- ✅ MIT License
- ✅ 完整的技术文档在 `docs/`

## 后续优化（可选）

1. **添加 Topics**（提高可发现性）
   - 在 GitHub 仓库页面点 "Add topics"
   - 推荐: `robotics`, `reinforcement-learning`, `model-based-rl`, `manipulation`, `robosuite`, `self-improvement`, `llm`, `mujoco`

2. **About 描述**
   填写: "Self-iterating grasp agent that learns from failure via LLM-driven parameter refinement. 75% solve rate with full ablation logs."

3. **固定 README 里的图片**
   GitHub 会自动显示，但如果想确保图片路径正确：
   - 推送后检查 README 显示是否正常
   - 如果 GIF 太大加载慢，可以用 `<img src="..." loading="lazy"/>`

## 常见问题

**Q: 推送时显示 "Permission denied (publickey)"**
A: 需要配置 SSH key，参考 https://docs.github.com/zh/authentication/connecting-to-github-with-ssh

**Q: 能不能推到已有仓库？**
A: 可以，但确保该仓库是空的或你愿意覆盖它。用 `git push -f origin main` 强制推送。

**Q: 想修改后再推一次？**
A: 修改文件 → `git add .` → `git commit -m "your message"` → `git push`

---

## ✅ 检查清单（推送前最后确认）

- [ ] 代码里没有 API key / 密码（已确认无）
- [ ] 没有你的私人文件（论文/姓名/学号，已确认无）
- [ ] README 里的图片路径正确（`assets/demo.gif`）
- [ ] requirements.txt 版本合理
- [ ] 代码能正常导入（已验证）

全部通过 → 可以放心推送 🎉
