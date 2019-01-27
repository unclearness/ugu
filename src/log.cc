/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "include/log.h"

#include <stdarg.h>
#include <stdio.h>

// multi line macro with do {...} while (0) guard
#define PRINT_MACRO       \
  do {                    \
    va_list va;           \
    va_start(va, format); \
    vprintf(format, va);  \
    va_end(va);           \
  } while (0)

namespace {
currender::LogLevel g_log_level_ = currender::LogLevel::kError;
}

namespace currender {

void set_log_level(LogLevel level) { g_log_level_ = level; }

LogLevel get_log_level() { return g_log_level_; }

void LOGD(const char *format, ...) {
  if (LogLevel::kDebug <= g_log_level_) {
    PRINT_MACRO;
  }
}

void LOGI(const char *format, ...) {
  if (LogLevel::kInfo <= g_log_level_) {
    PRINT_MACRO;
  }
}

void LOGW(const char *format, ...) {
  if (LogLevel::kWarning <= g_log_level_) {
    PRINT_MACRO;
  }
}
void LOGE(const char *format, ...) {
  if (LogLevel::kError <= g_log_level_) {
    PRINT_MACRO;
  }
}

}  // namespace currender
