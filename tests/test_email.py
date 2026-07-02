
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL_TO = "saishanghlang20122002@gmail.com"
EMAIL_FROM = "6631503129@lamduan.mfu.ac.th"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = "6631503129@lamduan.mfu.ac.th"
SMTP_PASS = "dlfo psst iyol jicj"

try:
    print("📧 Testing email notification...")
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = "✅ Test Email from LULC Pipeline"
    
    body = "This is a test email to verify the email notification system is working correctly!"
    msg.attach(MIMEText(body, 'plain'))
    
    if SMTP_PORT == 465:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    else:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
    
    server.login(SMTP_USER, SMTP_PASS)
    server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    server.close()
    print("✅ Email sent successfully!")
except Exception as e:
    print(f"❌ Failed to send email: {e}")
    import traceback
    traceback.print_exc()
