# Jekyll development Dockerfile using official GHA container
FROM ghcr.io/actions/jekyll-build-pages:latest

# Set working directory
WORKDIR /srv/jekyll

# Copy the site source
COPY ./docs .

# Expose port for Jekyll server
EXPOSE 4000

# Default command to run Jekyll server
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload", "--force_polling"]
