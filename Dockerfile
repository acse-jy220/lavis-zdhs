From pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

USER root

WORKDIR /home/LAVIS

ENV app_name="visual_chat" \
    HW_OBS_AK=GL1XBYLXHMCJYPWVWTP7 \
    HW_OBS_SK=m7vIhvIQ9W27So3jCVzuFTyqru67BQuK7xYp10lG \
    HW_OBS_SERVER=https://obs.cn-central-221.ovaijisuan.com \
    DEVICE_ID=0

RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

RUN chmod 777 -R /home/LAVIS

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["sh run_scripts/zdtc/server_blip2_stage2.sh"]