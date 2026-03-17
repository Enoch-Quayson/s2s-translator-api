-- Migration: add password_hash column to users table
-- Run this if you already applied the original schema.sql

ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
