"""
Comprehensive Monitoring System
Handles system health monitoring, performance tracking, and alerting
"""
import os
import logging
import time
import psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json
import requests
from config import Config

class MonitoringSystem:
    """Comprehensive system monitoring and alerting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        
        # Monitoring state
        self.system_metrics = {
            'start_time': datetime.now(timezone.utc),
            'last_check': None,
            'errors_count': 0,
            'warnings_count': 0
        }
        
        # Alert configuration
        self.alert_recipients = self._get_alert_recipients()
        self.smtp_config = self._get_smtp_config()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,          # 80% CPU usage
            'memory_usage': 80.0,       # 80% memory usage
            'disk_usage': 90.0,         # 90% disk usage
            'response_time': 5000,      # 5 seconds response time
            'error_rate': 5.0,          # 5% error rate
            'data_freshness': 3600      # 1 hour data freshness
        }
        
        # Initialize monitoring
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize monitoring system"""
        try:
            # Create monitoring directory
            monitor_dir = os.path.join(self.config.LOG_DIR, 'monitoring')
            os.makedirs(monitor_dir, exist_ok=True)
            
            # Initialize metrics storage
            self.metrics_file = os.path.join(monitor_dir, 'metrics.json')
            self.alerts_file = os.path.join(monitor_dir, 'alerts.log')
            
            self.logger.info("Monitoring system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing monitoring system: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.logger.info("Starting system monitoring")
        
        # Start monitoring loop in separate thread
        import threading
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        # Start health check server
        health_thread = threading.Thread(target=self._start_health_server, daemon=True)
        health_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                
                # Combine metrics
                all_metrics = {**system_metrics, **app_metrics}
                
                # Check thresholds and generate alerts
                alerts = self._check_thresholds(all_metrics)
                
                # Store metrics
                self._store_metrics(all_metrics)
                
                # Send alerts if any
                if alerts:
                    self._send_alerts(alerts)
                
                # Update last check time
                self.system_metrics['last_check'] = datetime.now(timezone.utc)
                
                # Sleep until next check
                time.sleep(self.config.MONITORING_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _collect_system_metrics(self) -> Dict:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'process_memory_mb': process_memory,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'system_health': self._calculate_system_health(cpu_usage, memory_usage, disk_usage)
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def _collect_application_metrics(self) -> Dict:
        """Collect application-specific metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'trading_active': self._is_trading_active(),
                'last_trade_time': self._get_last_trade_time(),
                'open_positions': self._get_open_positions_count(),
                'portfolio_value': self._get_portfolio_value(),
                'model_status': self._get_model_status(),
                'data_status': self._get_data_status()
            }
            
            # Calculate error rate
            error_rate = self._calculate_error_rate()
            metrics['error_rate'] = error_rate
            
            # Calculate data freshness
            data_freshness = self._calculate_data_freshness()
            metrics['data_freshness_seconds'] = data_freshness
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    def _calculate_system_health(self, cpu: float, memory: float, disk: float) -> str:
        """Calculate overall system health status"""
        try:
            if cpu > 90 or memory > 90 or disk > 95:
                return 'critical'
            elif cpu > 80 or memory > 80 or disk > 90:
                return 'warning'
            else:
                return 'healthy'
                
        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
            return 'unknown'
    
    def _is_trading_active(self) -> bool:
        """Check if trading system is active"""
        # This would check the trading system status
        # For now, return True during market hours
        now = datetime.now(timezone.utc)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=17, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _get_last_trade_time(self) -> Optional[str]:
        """Get timestamp of last trade"""
        # This would query the trading system
        # For now, return current time
        return datetime.now(timezone.utc).isoformat()
    
    def _get_open_positions_count(self) -> int:
        """Get count of open positions"""
        # This would query the portfolio manager
        # For now, return random value
        return 0
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # This would query the portfolio manager
        # For now, return initial capital
        return self.config.INITIAL_CAPITAL
    
    def _get_model_status(self) -> Dict:
        """Get AI model status"""
        try:
            # Check model version status
            from ai.model_versioning import model_manager
            
            latest_model = model_manager.get_latest_model_version()
            
            if latest_model:
                return {
                    'status': 'ok',
                    'version': latest_model['version'],
                    'last_updated': latest_model['created_at']
                }
            else:
                return {
                    'status': 'no_model',
                    'message': 'No models available'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_data_status(self) -> Dict:
        """Get data pipeline status"""
        try:
            from data.automated_updater import data_updater
            
            status = data_updater.get_data_status()
            
            # Summarize status
            total_pairs = len(status)
            ok_pairs = sum(1 for s in status.values() if s.get('status') == 'OK')
            
            return {
                'status': 'ok' if ok_pairs == total_pairs else 'warning',
                'total_pairs': total_pairs,
                'ok_pairs': ok_pairs,
                'last_update': max([s.get('last_date', '') for s in status.values() if s.get('last_date')], default='')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _calculate_error_rate(self) -> float:
        """Calculate system error rate"""
        try:
            # This would calculate from log files
            # For now, return 0%
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating error rate: {e}")
            return 0.0
    
    def _calculate_data_freshness(self) -> int:
        """Calculate data freshness in seconds"""
        try:
            # Check when data was last updated
            from data.automated_updater import data_updater
            
            status = data_updater.get_data_status()
            
            if not status:
                return 999999  # No data available
            
            # Find most recent update
            latest_date = None
            for s in status.values():
                if s.get('last_date'):
                    try:
                        date = datetime.strptime(s['last_date'], '%Y-%m-%d')
                        if latest_date is None or date > latest_date:
                            latest_date = date
                    except Exception as e:
                        import logging; logging.warning(f"Suppressed at core/monitoring_system.py:297: {e}")
                        continue
            
            if latest_date is None:
                return 999999
            
            # Calculate freshness
            now = datetime.now(timezone.utc)
            freshness = (now - latest_date).total_seconds()
            
            return int(freshness)
            
        except Exception as e:
            self.logger.error(f"Error calculating data freshness: {e}")
            return 999999
    
    def _check_thresholds(self, metrics: Dict) -> List[Dict]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        try:
            # System metrics checks
            if metrics.get('cpu_usage', 0) > self.thresholds['cpu_usage']:
                alerts.append({
                    'type': 'system',
                    'severity': 'warning',
                    'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                    'metric': 'cpu_usage',
                    'value': metrics['cpu_usage'],
                    'threshold': self.thresholds['cpu_usage']
                })
            
            if metrics.get('memory_usage', 0) > self.thresholds['memory_usage']:
                alerts.append({
                    'type': 'system',
                    'severity': 'warning',
                    'message': f"High memory usage: {metrics['memory_usage']:.1f}%",
                    'metric': 'memory_usage',
                    'value': metrics['memory_usage'],
                    'threshold': self.thresholds['memory_usage']
                })
            
            # Application metrics checks
            if metrics.get('error_rate', 0) > self.thresholds['error_rate']:
                alerts.append({
                    'type': 'application',
                    'severity': 'critical',
                    'message': f"High error rate: {metrics['error_rate']:.1f}%",
                    'metric': 'error_rate',
                    'value': metrics['error_rate'],
                    'threshold': self.thresholds['error_rate']
                })
            
            if metrics.get('data_freshness_seconds', 999999) > self.thresholds['data_freshness']:
                alerts.append({
                    'type': 'data',
                    'severity': 'warning',
                    'message': f"Stale data detected: {metrics['data_freshness_seconds']} seconds old",
                    'metric': 'data_freshness',
                    'value': metrics['data_freshness_seconds'],
                    'threshold': self.thresholds['data_freshness']
                })
            
            # Model status check
            model_status = metrics.get('model_status', {})
            if model_status.get('status') == 'error':
                alerts.append({
                    'type': 'model',
                    'severity': 'critical',
                    'message': f"Model error: {model_status.get('message', 'Unknown error')}",
                    'metric': 'model_status'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking thresholds: {e}")
            return []
    
    def _store_metrics(self, metrics: Dict):
        """Store metrics for historical analysis"""
        try:
            # Load existing metrics
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            # Add new metrics
            all_metrics.append(metrics)
            
            # Keep only last 1000 entries
            if len(all_metrics) > 1000:
                all_metrics = all_metrics[-1000:]
            
            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error storing metrics: {e}")
    
    def _send_alerts(self, alerts: List[Dict]):
        """Send alerts to configured recipients"""
        try:
            for alert in alerts:
                # Log alert
                alert_msg = f"[{alert['severity'].upper()}] {alert['message']}"
                self.logger.warning(alert_msg)
                
                # Write to alert log
                with open(self.alerts_file, 'a') as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()} - {alert_msg}\n")
                
                # Send email alert for critical issues
                if alert['severity'] == 'critical':
                    self._send_email_alert(alert)
                
                # Send webhook alert if configured
                self._send_webhook_alert(alert)
        
        except Exception as e:
            self.logger.error(f"Error sending alerts: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert"""
        try:
            if not self.smtp_config or not self.alert_recipients:
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email', 'forex_ai@monitoring.com')
            msg['To'] = ', '.join(self.alert_recipients)
            msg['Subject'] = f"Forex AI Alert: {alert['type']} - {alert['severity']}"
            
            # Email body
            body = f"""
            <html>
            <body>
            <h2>Forex AI Trading System Alert</h2>
            <p><strong>Type:</strong> {alert['type']}</p>
            <p><strong>Severity:</strong> {alert['severity']}</p>
            <p><strong>Message:</strong> {alert['message']}</p>
            <p><strong>Time:</strong> {datetime.now(timezone.utc).isoformat()}</p>
            <p><strong>Metric:</strong> {alert.get('metric', 'N/A')}</p>
            <p><strong>Value:</strong> {alert.get('value', 'N/A')}</p>
            <p><strong>Threshold:</strong> {alert.get('threshold', 'N/A')}</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['username'], self.smtp_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent to {self.alert_recipients}")
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _send_webhook_alert(self, alert: Dict):
        """Send webhook alert (e.g., to Slack or Teams)"""
        try:
            webhook_url = os.environ.get('ALERT_WEBHOOK_URL')
            
            if not webhook_url:
                return
            
            # Prepare payload
            payload = {
                'text': f"🚨 *Forex AI Alert*\n*Type:* {alert['type']}\n*Severity:* {alert['severity']}\n*Message:* {alert['message']}\n*Time:* {datetime.now(timezone.utc).isoformat()}"
            }
            
            # Send webhook
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("Webhook alert sent successfully")
            else:
                self.logger.warning(f"Webhook alert failed with status: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _get_alert_recipients(self) -> List[str]:
        """Get list of alert recipients"""
        # This would normally come from configuration
        # For now, return empty list
        return []
    
    def _get_smtp_config(self) -> Dict:
        """Get SMTP configuration"""
        # This would normally come from environment variables
        # For now, return empty dict
        return {}
    
    def _start_health_server(self):
        """Start simple HTTP health check server"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            
            class HealthHandler(BaseHTTPRequestHandler):
                def __init__(self, monitoring_system):
                    self.monitoring_system = monitoring_system
                    super().__init__()
                
                def do_GET(self):
                    if self.path == '/health':
                        # Get current metrics
                        metrics = self.monitoring_system._collect_system_metrics()
                        app_metrics = self.monitoring_system._collect_application_metrics()
                        
                        all_metrics = {**metrics, **app_metrics}
                        
                        # Determine health status
                        health_status = all_metrics.get('system_health', 'unknown')
                        
                        # Prepare response
                        response = {
                            'status': health_status,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'uptime': str(datetime.now(timezone.utc) - self.monitoring_system.system_metrics['start_time']),
                            'metrics': all_metrics
                        }
                        
                        # Send response
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(response, indent=2).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    # Suppress default logging
                    pass
            
            # Start server
            server = HTTPServer(('localhost', 8080), lambda *args: HealthHandler(self)(*args))
            self.logger.info("Health check server started on port 8080")
            server.serve_forever()
            
        except Exception as e:
            self.logger.error(f"Error starting health server: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            # Collect current metrics
            system_metrics = self._collect_system_metrics()
            app_metrics = self._collect_application_metrics()
            
            # Combine metrics
            all_metrics = {**system_metrics, **app_metrics}
            
            # Get recent alerts
            recent_alerts = self._get_recent_alerts()
            
            # Get uptime
            uptime = datetime.now(timezone.utc) - self.system_metrics['start_time']
            
            return {
                'status': all_metrics.get('system_health', 'unknown'),
                'uptime': str(uptime),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': all_metrics,
                'recent_alerts': recent_alerts,
                'system_info': {
                    'start_time': self.system_metrics['start_time'].isoformat(),
                    'last_check': self.system_metrics['last_check'].isoformat() if self.system_metrics['last_check'] else None,
                    'errors_count': self.system_metrics['errors_count'],
                    'warnings_count': self.system_metrics['warnings_count']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts from log file"""
        try:
            if not os.path.exists(self.alerts_file):
                return []
            
            # Read alert log file
            with open(self.alerts_file, 'r') as f:
                lines = f.readlines()
            
            # Parse recent alerts
            recent_alerts = []
            
            for line in lines[-limit:]:
                try:
                    # Parse alert line
                    parts = line.strip().split(' - ', 1)
                    if len(parts) == 2:
                        timestamp_str, message = parts
                        timestamp = datetime.fromisoformat(timestamp_str)
                        
                        # Parse severity and type from message
                        if message.startswith('[CRITICAL]'):
                            severity = 'critical'
                            msg = message[11:]
                        elif message.startswith('[WARNING]'):
                            severity = 'warning'
                            msg = message[9:]
                        else:
                            severity = 'info'
                            msg = message
                        
                        recent_alerts.append({
                            'timestamp': timestamp.isoformat(),
                            'severity': severity,
                            'message': msg
                        })
                except Exception as e:
                    import logging; logging.warning(f"Suppressed at core/monitoring_system.py:621: {e}")
                    continue
            
            return recent_alerts
            
        except Exception as e:
            self.logger.error(f"Error getting recent alerts: {e}")
            return []

# Singleton instance
monitoring_system = MonitoringSystem()