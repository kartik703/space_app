#!/usr/bin/env python3
"""
Real-Time Notification System
Handles email, SMS, Slack, and webhook notifications for space weather alerts
"""

import smtplib
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import asyncio
import websockets
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#space-weather-alerts"
    
    webhook_enabled: bool = False
    webhook_urls: List[str] = None
    
    sms_enabled: bool = False
    sms_api_key: str = ""
    sms_numbers: List[str] = None

class NotificationSystem:
    """Comprehensive notification system for space weather alerts"""
    
    def __init__(self, config_file: str = "config/notifications.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.websocket_clients = set()
        
    def _load_config(self) -> NotificationConfig:
        """Load notification configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return NotificationConfig(**config_data)
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Return default config
        return NotificationConfig()
    
    def save_config(self):
        """Save current configuration"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    async def send_alert_notification(self, alert: Dict):
        """Send alert through all enabled notification channels"""
        tasks = []
        
        if self.config.email_enabled:
            tasks.append(self._send_email_alert(alert))
        
        if self.config.slack_enabled:
            tasks.append(self._send_slack_alert(alert))
        
        if self.config.webhook_enabled:
            tasks.append(self._send_webhook_alert(alert))
        
        if self.config.sms_enabled:
            tasks.append(self._send_sms_alert(alert))
        
        # Send to WebSocket clients
        tasks.append(self._send_websocket_alert(alert))
        
        # Execute all notifications
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Notification {i} failed: {result}")
    
    async def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        if not self.config.email_recipients:
            return
        
        try:
            subject = f"🚨 {alert['severity']} Space Weather Alert"
            
            # Create HTML email
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'#f44336' if alert['severity'] == 'EXTREME' else '#ff9800'};">
                    🚨 {alert['severity']} Space Weather Alert
                </h2>
                
                <p><strong>Parameter:</strong> {alert['parameter']}</p>
                <p><strong>Current Value:</strong> {alert['current_value']:.2f}</p>
                <p><strong>Threshold:</strong> {alert['threshold_value']:.2f}</p>
                <p><strong>Description:</strong> {alert['description']}</p>
                <p><strong>Recommendation:</strong> {alert['recommendation']}</p>
                
                <hr>
                <p><small>Alert generated at: {alert['timestamp']}</small></p>
                <p><small>Space Weather Monitoring System</small></p>
            </body>
            </html>
            """
            
            # Send email
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            
            html_part = MimeText(html_body, 'html')
            msg.attach(html_part)
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert['parameter']}")
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    async def _send_slack_alert(self, alert: Dict):
        """Send Slack alert"""
        if not self.config.slack_webhook_url:
            return
        
        try:
            # Color based on severity
            colors = {
                'LOW': '#36a64f',
                'MEDIUM': '#ffeb3b',
                'HIGH': '#ff9800',
                'EXTREME': '#f44336'
            }
            
            payload = {
                "channel": self.config.slack_channel,
                "username": "Space Weather Bot",
                "icon_emoji": ":satellite:",
                "attachments": [
                    {
                        "color": colors.get(alert['severity'], '#999999'),
                        "title": f"🚨 {alert['severity']} Space Weather Alert",
                        "text": alert['description'],
                        "fields": [
                            {
                                "title": "Parameter",
                                "value": alert['parameter'],
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert['current_value']:.2f}",
                                "short": True
                            },
                            {
                                "title": "Recommendation",
                                "value": alert['recommendation'],
                                "short": False
                            }
                        ],
                        "footer": "Space Weather Monitoring",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for {alert['parameter']}")
            
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Dict):
        """Send webhook notifications"""
        if not self.config.webhook_urls:
            return
        
        for webhook_url in self.config.webhook_urls:
            try:
                payload = {
                    "type": "space_weather_alert",
                    "timestamp": datetime.now().isoformat(),
                    "alert": alert
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
                logger.info(f"Webhook alert sent to {webhook_url}")
                
            except Exception as e:
                logger.error(f"Webhook alert failed for {webhook_url}: {e}")
    
    async def _send_sms_alert(self, alert: Dict):
        """Send SMS alert (placeholder - requires SMS service)"""
        if not self.config.sms_numbers:
            return
        
        try:
            # This is a placeholder - you would integrate with Twilio, AWS SNS, etc.
            message = f"🚨 {alert['severity']} Alert: {alert['description'][:100]}..."
            
            logger.info(f"SMS alert would be sent: {message}")
            # TODO: Implement actual SMS sending
            
        except Exception as e:
            logger.error(f"SMS alert failed: {e}")
    
    async def _send_websocket_alert(self, alert: Dict):
        """Send real-time WebSocket alert"""
        if not self.websocket_clients:
            return
        
        try:
            message = json.dumps({
                "type": "alert",
                "data": alert,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send to all connected clients
            disconnected = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected
            
            logger.info(f"WebSocket alert sent to {len(self.websocket_clients)} clients")
            
        except Exception as e:
            logger.error(f"WebSocket alert failed: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected. Total: {len(self.websocket_clients)}")
        
        try:
            # Send welcome message
            welcome = {
                "type": "welcome",
                "message": "Connected to Space Weather Alert System",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome))
            
            # Keep connection alive
            async for message in websocket:
                # Echo received messages (for testing)
                response = {
                    "type": "echo",
                    "received": message,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.websocket_clients)}")
    
    def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time notifications"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_server = websockets.serve(self.websocket_handler, host, port)
            loop.run_until_complete(start_server)
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        return thread

class NotificationTester:
    """Test notification system"""
    
    def __init__(self, notification_system: NotificationSystem):
        self.notification_system = notification_system
    
    async def test_all_notifications(self):
        """Test all notification channels"""
        test_alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH',
            'parameter': 'kp_index',
            'current_value': 7.5,
            'threshold_value': 7.0,
            'description': 'Geomagnetic activity HIGH - Kp index at 7.5',
            'recommendation': 'HIGH RISK: Monitor closely, prepare contingency measures'
        }
        
        print("🧪 Testing notification system...")
        print(f"📧 Email enabled: {self.notification_system.config.email_enabled}")
        print(f"💬 Slack enabled: {self.notification_system.config.slack_enabled}")
        print(f"🔗 Webhook enabled: {self.notification_system.config.webhook_enabled}")
        print(f"📱 SMS enabled: {self.notification_system.config.sms_enabled}")
        
        await self.notification_system.send_alert_notification(test_alert)
        print("✅ Test notifications sent!")

def main():
    """Demo the notification system"""
    print("🔔 SPACE WEATHER NOTIFICATION SYSTEM")
    print("=" * 45)
    
    # Create notification system
    notif_system = NotificationSystem()
    
    # Start WebSocket server
    ws_thread = notif_system.start_websocket_server()
    
    # Test notifications
    async def run_test():
        tester = NotificationTester(notif_system)
        await tester.test_all_notifications()
    
    # Run test
    asyncio.run(run_test())
    
    print("\n📊 Configuration:")
    print(f"📧 Email: {'✅' if notif_system.config.email_enabled else '❌'}")
    print(f"💬 Slack: {'✅' if notif_system.config.slack_enabled else '❌'}")
    print(f"🔗 Webhook: {'✅' if notif_system.config.webhook_enabled else '❌'}")
    print(f"📱 SMS: {'✅' if notif_system.config.sms_enabled else '❌'}")
    print(f"🌐 WebSocket: ✅ Running on ws://localhost:8765")
    
    print("\n💡 To configure notifications:")
    print("1. Edit config/notifications.json")
    print("2. Enable desired notification channels")
    print("3. Add your credentials and endpoints")
    
    try:
        input("\nPress Enter to stop WebSocket server...")
    except KeyboardInterrupt:
        pass
    
    print("✅ Notification system demo complete!")

if __name__ == "__main__":
    main()