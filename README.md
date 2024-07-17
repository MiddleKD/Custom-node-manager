# 🎨 Custom-nodes-manager
comfy ui의 custom nodes를 git 폴더를 제거하고 복사해서 저장합니다.
custom nodes의 버전 관리를 위한 코드입니다.

## 🚀 Introduction
- ComfyUI의 사용자 정의 노드는 오픈 소스로 제공되며 지속적으로 업데이트됩니다.
- 따라서 노드 간 충돌이 발생할 경우, 개발자는 직접 노드 코드를 수정해야 합니다.
- 이 저장소는 ComfyUI의 사용자 정의 노드를 해당 폴더 구조 그대로 보관하는 용도로 사용됩니다.
- ComfyUI 기반 서비스를 배포할 때 유용하며, 정적이면서도 조절 가능한 사용자 정의 노드 관리가 가능합니다.

## 📥 Install
```bash
# 먼저 GitHub 등에서 해당 리포지토리를 포크한 후,
# 포크한 본인의 리포지토리를 클론합니다.
git clone {your_forked_repository}
```
## 🖥 How to use
1. ComfyUI의 custom_nodes 복사
    ```bash
    cd Custom-nodes-manager
    python3 cp_custom_nodes.py --src {your_comfyui_custom_nodes_dir_path}
    ```
2. 원격 저장소에 저장
    ```bash
    git add .
    git commit -m "Update: custom_nodes"
    git push
    ```
3. 배포된 서버에 저장
    ```bash
    # server(AWS, NCP etc.)의 ComfyUI 경로로 이동합니다.
    cd {your_comfyui_path}

    # 기존 custom_nodes dir을 제거합니다.
    rm -rf custom_nodes

    # 본인의 리포지토리를 클론하고, 저장된 custom_nodes를 옮깁니다.
    git clone {your_forked_repository}
	mv Custom-nodes-manager/custom_nodes/ custom_nodes
	rm -rf Custom-nodes-manager
    ```