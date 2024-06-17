# Remove_bg_aws
remove bg AI모델 AWS lambda 및 EC2 환경 구축용

## Dockerfile
- requirements 파일은 필요하지 않습니다. 참고로만 사용해주세요.
- lambda용, ec2용 Dockerfile과 .dockerignore가 있습니다.

## USAGE

*AWS lambda ECR*
- only cpu
*usage*
- ./docker_file/Dockerfile(lambda) => ./Dockerfile
- ./docker_file/.dockerignore(lambda) => ./.dockerignore
- docker build -t remove_bg:lambda
- docker tag remove_bg:lambda your_aws_container_id:tag
- docker push your_aws_container_id:tag

*AWS EC2*
- avail gpu
- Flask api
*usage*
- ./docker_file/Dockerfile(ec2) => ./Dockerfile
- ./docker_file/.dockerignore(ec2) => ./.dockerignore
- docker build -t remove_bg:ec2
- docker tag remove_bg:lambda your_docker_user_name/remove_bg:lambda
- Connect docker in your EC2
*API*
- awsurl/sample => easy test
- awsurl/post/bs64
- awsurl/post/url
