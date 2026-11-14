# The spatial agent: the dora-world-agent node and the world-agent console, with a pinned Claude Code CLI.
# The workspace depends on perception-wire in dora-depth-perception, so build from the nodes directory:
#   docker build -f dora-world-agent/Dockerfile -t dora-world-agent .
# Credentials come at run time (see the README); none are in the image.

FROM rust:1.90.0-bookworm AS build
WORKDIR /src
COPY dora-depth-perception dora-depth-perception
COPY dora-world-agent dora-world-agent
WORKDIR /src/dora-world-agent
RUN cargo build --release --locked --bin world-agent --bin dora-world-agent

FROM node:22-bookworm-slim
# The CLI version the agent was tested against; the stream-json control protocol changes between versions.
ARG CLAUDE_CODE_VERSION=2.1.325
RUN npm install --global "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" && npm cache clean --force
ENV DISABLE_AUTOUPDATER=1
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /var/lib/spatial-agent/backup /etc/spatial-agent /home/agent/.claude \
    && chown -R agent:agent /var/lib/spatial-agent /home/agent/.claude
COPY --from=build /src/dora-world-agent/target/release/world-agent /src/dora-world-agent/target/release/dora-world-agent /usr/local/bin/
COPY dora-world-agent/world-agent/spatial-agent.container.toml /etc/spatial-agent/settings.toml
# The node reads its settings from here; the console takes the same file as its argument.
ENV SPATIAL_AGENT_CONFIG=/etc/spatial-agent/settings.toml
USER agent
WORKDIR /var/lib/spatial-agent
ENTRYPOINT ["world-agent", "/etc/spatial-agent/settings.toml"]
