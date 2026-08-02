# InferFlow

A model inference service built from raw sockets — no frameworks. Implements I/O multiplexing, a multi-process worker pool, a Redis-backed request queue, and load balancing, containerized with Docker and run as a systemd daemon.

## Status

🚧 In progress — Week 1: model + basic socket server

## Roadmap

- [ ] Week 1 — ONNX model + blocking socket server
- [ ] Week 2 — epoll (I/O multiplexing) + multi-process worker pool + Redis queue
- [ ] Week 3 — Load balancer + Locust benchmarking
- [ ] Week 4 — Docker Compose + systemd daemon + documentation

## Tech

Python · ONNX Runtime · Redis · Docker · systemd · Locust
