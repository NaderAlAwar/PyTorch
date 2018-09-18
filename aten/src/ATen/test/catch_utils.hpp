#pragma once

#define CATCH_CONFIG_PREFIX_ALL
#include <catch.hpp>

// CATCH_REQUIRE_THROWS is not defined identically to REQUIRE_THROWS and causes warning;
// define our own version that doesn't warn.
INTERNAL_CATCH_THROWS( "_CATCH_REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__ )
