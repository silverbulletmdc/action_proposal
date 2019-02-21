import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 第三方 SMTP 服务
mail_host = "*"  # 设置服务器
mail_user = "*"  # 用户名
mail_pass = "*"  # 口令

sender = '*'
receivers = ['*']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱


def send_mail(content):
    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header("vipl server", 'utf-8')  # 发送者
    message['To'] = Header("*", 'utf-8')  # 接收者

    subject = content
    message['Subject'] = Header(subject, 'utf-8')

    smtp_obj = smtplib.SMTP()
    smtp_obj.connect(mail_host, 25)
    smtp_obj.login(mail_user, mail_pass)
    smtp_obj.sendmail(sender, receivers, message.as_string())
