#!/usr/bin/env bash
set -e

if [[ -z "$HOST_UID" ]]; then
	    echo "ERROR: please set HOST_UID" >&2
	        exit 1
fi
if [[ -z "$HOST_GID" ]]; then
	    echo "ERROR: please set HOST_GID" >&2
	        exit 1
fi

groupmod --gid "$HOST_GID" docker-user
usermod --uid "$HOST_UID" docker-user
chown -R $USER:$USER /opt/venv
echo "Password: Docker!"

if [[ $# -gt 0 ]]; then
	    exec sudo -u docker-user -- "$@"
    else
	        exec sudo -u docker-user -- bash
fi