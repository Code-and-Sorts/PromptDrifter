"""
Centralized HTTP client management for PromptDrifter adapters.
Provides connection pooling and efficient resource management.
"""

import asyncio
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx


class HTTPClientManager:
    """
    Manages shared HTTP clients with connection pooling for optimal performance.

    Benefits:
    - Connection reuse across adapters
    - Reduced connection overhead
    - Configurable timeouts and limits
    - Automatic cleanup
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout_seconds: float = 30.0,
        max_retries: int = 3
    ):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Cache of HTTP clients by base URL
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()

    async def get_client(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> httpx.AsyncClient:
        """
        Get or create an HTTP client for the given base URL.

        Args:
            base_url: Base URL for the client
            headers: Default headers to include

        Returns:
            Configured AsyncClient instance
        """
        # Normalize base URL for consistent caching
        parsed = urlparse(base_url)
        cache_key = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock:
            if cache_key not in self._clients:
                # Configure connection limits for optimal performance
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_keepalive_connections
                )

                # Configure timeout settings
                timeout = httpx.Timeout(
                    connect=10.0,  # Connection timeout
                    read=self.timeout_seconds,  # Read timeout
                    write=self.timeout_seconds,  # Write timeout
                    pool=self.timeout_seconds   # Pool timeout
                )

                # Create client with connection pooling
                client = httpx.AsyncClient(
                    base_url=base_url,
                    headers=headers or {},
                    limits=limits,
                    timeout=timeout,
                    follow_redirects=True
                )

                self._clients[cache_key] = client

            return self._clients[cache_key]

    async def close_all(self):
        """Close all HTTP clients and clean up connections."""
        async with self._lock:
            close_tasks = []

            for client in self._clients.values():
                close_tasks.append(client.aclose())

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            self._clients.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about managed HTTP clients."""
        return {
            "active_clients": len(self._clients),
            "base_urls": list(self._clients.keys()),
            "max_connections": self.max_connections,
            "max_keepalive": self.max_keepalive_connections
        }


# Global HTTP client manager instance
_http_client_manager = None


def get_http_client_manager() -> HTTPClientManager:
    """Get the global HTTP client manager instance."""
    global _http_client_manager
    if _http_client_manager is None:
        _http_client_manager = HTTPClientManager()
    return _http_client_manager


async def get_shared_client(
    base_url: str,
    headers: Optional[Dict[str, str]] = None
) -> httpx.AsyncClient:
    """
    Convenience function to get a shared HTTP client.

    Args:
        base_url: Base URL for the client
        headers: Default headers to include

    Returns:
        Shared AsyncClient instance with connection pooling
    """
    manager = get_http_client_manager()
    return await manager.get_client(base_url, headers)
