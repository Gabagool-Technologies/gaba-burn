use crate::{Result, ServeError};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct SchedulerConfig {
    pub max_concurrent: usize,
    pub enable_priority: bool,
    pub timeout_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 100,
            enable_priority: true,
            timeout_ms: 5000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

pub struct ScheduledRequest {
    pub id: u64,
    pub priority: Priority,
    pub timestamp: Instant,
    pub deadline: Option<Instant>,
}

impl PartialEq for ScheduledRequest {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for ScheduledRequest {}

impl PartialOrd for ScheduledRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.timestamp.cmp(&self.timestamp),
            other => other,
        }
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

pub struct RequestScheduler {
    config: SchedulerConfig,
    queue: Arc<Mutex<BinaryHeap<ScheduledRequest>>>,
    active_count: Arc<Mutex<usize>>,
}

impl RequestScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            active_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn schedule(&self, id: u64, priority: Priority) -> Result<()> {
        let mut queue = self.queue.lock().unwrap();
        
        let request = ScheduledRequest {
            id,
            priority,
            timestamp: Instant::now(),
            deadline: None,
        };
        
        queue.push(request);
        Ok(())
    }

    pub fn schedule_with_deadline(&self, id: u64, priority: Priority, deadline: Instant) -> Result<()> {
        let mut queue = self.queue.lock().unwrap();
        
        let request = ScheduledRequest {
            id,
            priority,
            timestamp: Instant::now(),
            deadline: Some(deadline),
        };
        
        queue.push(request);
        Ok(())
    }

    pub fn next_request(&self) -> Option<ScheduledRequest> {
        let active = *self.active_count.lock().unwrap();
        if active >= self.config.max_concurrent {
            return None;
        }

        let mut queue = self.queue.lock().unwrap();
        
        while let Some(request) = queue.pop() {
            if let Some(deadline) = request.deadline {
                if Instant::now() > deadline {
                    continue;
                }
            }
            
            *self.active_count.lock().unwrap() += 1;
            return Some(request);
        }
        
        None
    }

    pub fn complete_request(&self, _id: u64) {
        let mut active = self.active_count.lock().unwrap();
        if *active > 0 {
            *active -= 1;
        }
    }

    pub fn queue_size(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn active_count(&self) -> usize {
        *self.active_count.lock().unwrap()
    }

    pub fn can_accept(&self) -> bool {
        self.active_count() < self.config.max_concurrent
    }

    pub fn clear_expired(&self) {
        let mut queue = self.queue.lock().unwrap();
        let now = Instant::now();
        
        let valid_requests: Vec<_> = queue
            .drain()
            .filter(|req| {
                req.deadline.map_or(true, |deadline| now <= deadline)
            })
            .collect();
        
        for req in valid_requests {
            queue.push(req);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = RequestScheduler::new(config);
        assert_eq!(scheduler.queue_size(), 0);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_schedule_request() {
        let config = SchedulerConfig::default();
        let scheduler = RequestScheduler::new(config);
        
        scheduler.schedule(1, Priority::Normal).unwrap();
        assert_eq!(scheduler.queue_size(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let config = SchedulerConfig::default();
        let scheduler = RequestScheduler::new(config);
        
        scheduler.schedule(1, Priority::Low).unwrap();
        scheduler.schedule(2, Priority::High).unwrap();
        scheduler.schedule(3, Priority::Normal).unwrap();
        
        let first = scheduler.next_request().unwrap();
        assert_eq!(first.id, 2);
        assert_eq!(first.priority, Priority::High);
    }

    #[test]
    fn test_max_concurrent() {
        let config = SchedulerConfig {
            max_concurrent: 2,
            ..Default::default()
        };
        let scheduler = RequestScheduler::new(config);
        
        scheduler.schedule(1, Priority::Normal).unwrap();
        scheduler.schedule(2, Priority::Normal).unwrap();
        scheduler.schedule(3, Priority::Normal).unwrap();
        
        let req1 = scheduler.next_request();
        let req2 = scheduler.next_request();
        let req3 = scheduler.next_request();
        
        assert!(req1.is_some());
        assert!(req2.is_some());
        assert!(req3.is_none());
        
        assert_eq!(scheduler.active_count(), 2);
    }

    #[test]
    fn test_complete_request() {
        let config = SchedulerConfig::default();
        let scheduler = RequestScheduler::new(config);
        
        scheduler.schedule(1, Priority::Normal).unwrap();
        let req = scheduler.next_request().unwrap();
        
        assert_eq!(scheduler.active_count(), 1);
        scheduler.complete_request(req.id);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_deadline_expiration() {
        let config = SchedulerConfig::default();
        let scheduler = RequestScheduler::new(config);
        
        let past_deadline = Instant::now() - Duration::from_secs(1);
        scheduler.schedule_with_deadline(1, Priority::Normal, past_deadline).unwrap();
        
        let req = scheduler.next_request();
        assert!(req.is_none());
    }

    #[test]
    fn test_can_accept() {
        let config = SchedulerConfig {
            max_concurrent: 1,
            ..Default::default()
        };
        let scheduler = RequestScheduler::new(config);
        
        assert!(scheduler.can_accept());
        
        scheduler.schedule(1, Priority::Normal).unwrap();
        scheduler.next_request();
        
        assert!(!scheduler.can_accept());
    }

    #[test]
    fn test_priority_comparison() {
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
        assert!(Priority::Critical > Priority::High);
    }
}
