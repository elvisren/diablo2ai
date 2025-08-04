import cv2

# 创建视频捕获对象（0=内置摄像头，1=外接摄像头）
cap = cv2.VideoCapture(1)
# 0: the iphone camera
# 1: the logipro camera

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法访问摄像头！")
    exit()

print("摄像头已打开 - 按 ESC 退出")

# 设置分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 高度

try:
    while True:
        # 读取当前帧
        ret, frame = cap.read()

        # 检查帧是否有效
        if not ret:
            print("无法获取帧，请检查连接")
            break

        # 水平翻转（镜像效果，可选）
        frame = cv2.flip(frame, 1)

        # 显示视频
        cv2.imshow('摄像头视频流', frame)

        # 按ESC键退出（等待1ms）
        if cv2.waitKey(1) == 27:  # ESC键的ASCII码
            break
finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出，释放资源")
